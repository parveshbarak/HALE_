use npyz::WriterBuilder;
use rustc_hash::FxHashMap as HashMap;
use std::fs::{create_dir_all, File};
use std::io::prelude::*;
use std::io::{BufWriter, Result};
use std::path::Path;

use crossbeam_channel::Sender;

use ndarray::{s, stack, Array, Array2, ArrayBase, ArrayViewMut1, Axis, Data, Ix2};
use ordered_float::OrderedFloat;

use crate::aligners::{CigarIter, CigarOp};
use crate::haec_io::HAECRecord;
use crate::correct::{prepare_examples, CorrectData, WindowExample};
use crate::overlaps::{Alignment, Strand};
use crate::pbars::PBarNotification;
use crate::windowing::{extract_windows, OverlapWindow};

pub(crate) const TOP_K: usize = 20;

const BASE_LOWER: [u8; 128] = [
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 97, 255, 99, 255, 255, 255, 103, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 116, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
];

const BASE_FORWARD: [u8; 128] = [
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 42, 255, 255,
    255, 255, 255, 255, 42, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 65, 255, 67, 255, 255, 255, 71, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 84, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 65, 255, 67, 255, 255, 255, 71, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 84, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
];

const ALIGNMENT_LEN_TH:usize = 4000;



// Heuristics to filter rows from based on alignment accuracy
// Heuristic 3:
    // The core idea of Heuristic Three is to limit the column-wise coverage of non-gap bases, ensuring that at no position (col), 
    // the number of aligned (non-gap) segments passing through it exceeds a threshold (top_k = 20).
        // It processes rows sequentially and maintains a running coverage count for each column.
        // For each new row, it attempts to include the row, or portions of it, only if doing so doesn't push the column coverage above top_k.
        // If including a segment would violate the coverage constraint, that segment is "chopped" (excluded), and the function looks for subsequent, valid segments in the same row.
        // Valid segments are those that can be included without exceeding the top_k coverage limit and are at least ALIGNMENT_LEN_TH bases long.
        // The final output arrays, filtered_bases and filtered_quals, contain the selected and chopped segments, 
        // where excluded regions (or non-selected columns) are filled with the gap character (b'.').
fn filter_rows_heuristic_three(bases: &Array2<u8>, quals: &Array2<u8>) -> (Array2<u8>, Array2<u8>) {
    let top_k = 20;
    let ncols = bases.ncols();
    let mut coverage = vec![0usize; ncols];

    let mut filtered_rows = Vec::new();
    filtered_rows.push((0, 0, ncols-1));
    // iterate over rows
    for row in 1..bases.nrows() {
        let mut is_chop = false;
        // create a vector to store <start,end index pairs> in each row where coverage const are not violated
        let mut valid_ranges = Vec::new();
        let mut start_idx = 0;
        let mut end_idx = 0;
        let mut no_chop_start = 0;
        let mut no_chop_end = ncols-1;
        let mut first_non_gap_found = false;
        {
            let row_view = bases.row(row);
            for (col, &val) in row_view.iter().enumerate() {
                if val != b'.' {
                    if !first_non_gap_found {
                        first_non_gap_found = true;
                        no_chop_start = col;
                        start_idx = col;
                        end_idx = col;
                    }
                    if coverage[col] + 1 > top_k {
                        is_chop = true;
                        if end_idx - start_idx >= ALIGNMENT_LEN_TH {
                            valid_ranges.push((start_idx, end_idx-1));
                        }
                        start_idx = col + 1;
                    }
                    end_idx += 1;
                } else {
                    if first_non_gap_found {
                        no_chop_end = col - 1;
                        break;
                    }
                }
            }
            if !is_chop {
                filtered_rows.push((row, no_chop_start, no_chop_end)); // keep the whole row
                // add 1 to all indices in range
                for i in no_chop_start..=no_chop_end {
                    coverage[i] += 1;
                }
            } else if is_chop && valid_ranges.len() > 0 {
                for (start, end) in valid_ranges {
                    filtered_rows.push((row, start, end));
                    for i in start..=end {
                        coverage[i] += 1;
                    }
                }
            }
        }
    }

    let mut filtered_bases = Array2::zeros((filtered_rows.len(), ncols));
    let mut filtered_quals = Array2::zeros((filtered_rows.len(), ncols));
    let mut row_idx = 0;
    for (row, start, end) in filtered_rows {
        for col in 0..ncols {
            if col >= start && col <= end {
                filtered_bases[(row_idx, col)] = bases[(row, col)];
                filtered_quals[(row_idx, col)] = quals[(row, col)];
            } else {
                filtered_bases[(row_idx, col)] = b'.';
            }
        }
        row_idx += 1;
    }

    (filtered_bases, filtered_quals)
}








fn get_max_ins_for_window(
    overlaps: &[OverlapWindow], // Sorted overlaps
    ovlps_cigar_map: &HashMap<u32, &Vec<u8>>,
    tid: u32,
    window_length: usize,
) -> Vec<u16> {
    let mut max_ins = vec![0; window_length];
    for ow in overlaps.iter() {
        let is_target = ow.overlap.tid == tid;
        if is_target {
            let mut tpos = ow.overlap.tstart as usize;
            // Handle cigar
            let qid = ow.overlap.return_other_id(tid);
            let cigar = ovlps_cigar_map.get(&qid).unwrap();
            //let cigar_len = ow.cigar_end_idx - ow.cigar_start_idx + 1;
            let cigar_iter = CigarIter::new(&cigar);

            cigar_iter.for_each(|(op, range)| {
                let l = match op {
                    CigarOp::Match(l) | CigarOp::Mismatch(l) | CigarOp::Deletion(l) => l as usize,
                    CigarOp::Insertion(l) => {
                        assert!(
                            tpos <= max_ins.len(),
                            "Length {} is bigger than the tseq {}. {} {} {} {:?} {:?}",
                            tpos,
                            max_ins.len(),
                            ow.cigar_start_offset,
                            op.get_length(),
                            std::str::from_utf8(cigar).unwrap(),
                            range,
                            ow.cigar_start_idx..ow.cigar_end_idx
                        );

                        max_ins[tpos - 1] = max_ins[tpos - 1].max(l as u16);
                        return;
                    }
                };
                tpos += l;
            });
        } else {
            let mut tpos = ow.overlap.qstart as usize;
            // Handle cigar
            let qid = ow.overlap.return_other_id(tid);
            let cigar = ovlps_cigar_map.get(&qid).unwrap();
            //let cigar_len = ow.cigar_end_idx - ow.cigar_start_idx + 1;
            let cigar_iter = CigarIter::new(&cigar);

            cigar_iter.for_each(|(op, range)| {
                let l = match op {
                    CigarOp::Match(l) | CigarOp::Mismatch(l) | CigarOp::Insertion(l) => l as usize,
                    CigarOp::Deletion(l) => {
                        assert!(
                            tpos <= max_ins.len(),
                            "Length {} is bigger than the tseq {}. {} {} {} {:?} {:?}",
                            tpos,
                            max_ins.len(),
                            ow.cigar_start_offset,
                            op.get_length(),
                            std::str::from_utf8(cigar).unwrap(),
                            range,
                            ow.cigar_start_idx..ow.cigar_end_idx
                        );

                        max_ins[tpos - 1] = max_ins[tpos - 1].max(l as u16);
                        return;
                    }
                };
                tpos += l;
            });
        }
    }

    max_ins
}




fn get_query_region(window: &OverlapWindow, tid: u32) -> (u32, u32) {
    let (qstart, qend) = if window.overlap.tid == tid {
        (window.overlap.qstart, window.overlap.qend)
    } else {
        (window.overlap.tstart, window.overlap.tend)
    };

    match window.overlap.strand {
        Strand::Forward => (qstart + window.qstart, qstart + window.qend),
        Strand::Reverse => (qend - window.qend, qend - window.qstart),
    }
}

fn get_features_for_ol_window(
    mut bases: ArrayViewMut1<'_, u8>,
    mut quals: ArrayViewMut1<'_, u8>,
    window: &OverlapWindow,
    cigar: &[u8],
    query: &HAECRecord,
    tid: u32,
    max_ins: &[u16],
    qbuffer: &mut [u8],
) {
    // Handle query sequence
    let (qstart, qend) = if window.overlap.tid == tid {
        (window.overlap.qstart, window.overlap.qend)
    } else {
        (window.overlap.tstart, window.overlap.tend)
    };

    let (tstart, tend) = if window.overlap.tid != tid {
        (window.overlap.qstart, window.overlap.qend)
    } else {
        (window.overlap.tstart, window.overlap.tend)
    };

    let is_target = window.overlap.tid == tid;

    let mut query_iter: Box<dyn DoubleEndedIterator<Item = (&u8, &u8)>> =
        match window.overlap.strand {
            Strand::Forward => {
                let range = qstart as usize..qend as usize;
                let qlen = qend as usize - qstart as usize;

                query.seq.get_subseq(range.clone(), qbuffer);
                let quals = &query.qual[range];

                Box::new(qbuffer[..qlen].iter().zip(quals))
            }
            Strand::Reverse => {
                let range = qstart as usize..qend as usize;
                let qlen = qend as usize - qstart as usize;

                query.seq.get_rc_subseq(range.clone(), qbuffer);
                let quals = &query.qual[range];

                Box::new(
                    qbuffer[..qlen]
                        .iter()
                        .zip(quals.iter().rev())
                        .map(|(b, q)| (&BASE_LOWER[*b as usize], q)),
                )
            }
        };
    //let mut query_iter = query_iter.skip(window.qstart as usize);

    // Number of cigars for the window
    // TODO get error when we calculate correct number for end -> (idx, 0)
    // Works for this expression but unecessarily iterates through (idx, 0)
    //let cigar_len = window.cigar_end_idx - window.cigar_start_idx + 1;
    //let cigar_end = cigar.len().min((window.cigar_end_idx + 1) as usize);

    // Handle cigar
    //let cigar = cigar[window.cigar_start_idx as usize..cigar_end].iter();
    let cigar = CigarIter::new(&cigar);

    // Get features
    let gap = if let Strand::Forward = window.overlap.strand {
        b'*'
    } else {
        b'#'
    };
    bases.fill(gap); // Initialize with gap token

    let mut tpos = tstart as usize; // position in the target read (excluding insertions)
    let mut idx = tstart as usize + max_ins[..tstart as usize].iter().map(|v| *v as usize).sum::<usize>(); // position in the features (including insertions)

    if idx > 0 {
        // No alignment at the start
        bases.slice_mut(s![..idx]).fill(b'.');
    }

    cigar.for_each(|(op, range)| {
        let mut l = match op {
            CigarOp::Match(l)
            | CigarOp::Mismatch(l)
            | CigarOp::Deletion(l)
            | CigarOp::Insertion(l) => l as usize,
        };

        // Write features
        if is_target {
            match op {
                CigarOp::Match(_) | CigarOp::Mismatch(_) => {
                    for i in 0..l {
                        let (base, qual) = query_iter
                            .next()
                            .expect("Base and its quality should be present.");
                        bases[idx] = *base;
                        quals[idx] = *qual;

                        idx += 1 + max_ins[tpos + i] as usize;
                    }

                    tpos += l;
                }
                CigarOp::Deletion(_) => {
                    for i in 0..l {
                        // No need to write gap, gap is already written
                        idx += 1 + max_ins[tpos + i] as usize;
                    }

                    tpos += l;
                }
                CigarOp::Insertion(_) => {
                    /*assert!(
                        max_ins[tpos - 1] as usize >= l,
                        "Insertion length is bigger than max_ins"
                    );*/

                    idx -= max_ins[tpos - 1] as usize; // Return to first insertion for the previous base
                    for i in 0..l {
                        let (base, qual) = query_iter
                            .next()
                            .expect("Base and its quality should be present.");

                        bases[idx + i] = *base;
                        quals[idx + i] = *qual;
                    }
                    idx += max_ins[tpos - 1] as usize; // Move back to the last base
                }
            }
        } else {
            match op {
                CigarOp::Match(_) | CigarOp::Mismatch(_) => {
                    for i in 0..l {
                        let (base, qual) = query_iter
                            .next()
                            .expect("Base and its quality should be present.");
                        bases[idx] = *base;
                        quals[idx] = *qual;

                        idx += 1 + max_ins[tpos + i] as usize;
                    }

                    tpos += l;
                }
                CigarOp::Insertion(_) => {
                    for i in 0..l {
                        // No need to write gap, gap is already written
                        idx += 1 + max_ins[tpos + i] as usize;
                    }

                    tpos += l;
                }
                CigarOp::Deletion(_) => {
                    /*assert!(
                        max_ins[tpos - 1] as usize >= l,
                        "Insertion length is bigger than max_ins"
                    );*/

                    idx -= max_ins[tpos - 1] as usize; // Return to first insertion for the previous base
                    for i in 0..l {
                        let (base, qual) = query_iter
                            .next()
                            .expect("Base and its quality should be present.");

                        bases[idx + i] = *base;
                        quals[idx + i] = *qual;
                    }
                    idx += max_ins[tpos - 1] as usize; // Move back to the last base
                }
            }
        }
    });

    if idx < bases.shape()[0] {
        // No alignment at the end
        bases.slice_mut(s![idx..]).fill(b'.');
    }
}

fn write_target_for_window(
    target: &HAECRecord,
    max_ins: &[u16],
    mut bases: ArrayViewMut1<'_, u8>,
    mut quals: ArrayViewMut1<'_, u8>,
    window_length: usize,
    tbuffer: &[u8],
) {
    bases.fill(b'*'); // Fill like forward

    let mut tpos = 0;
    tbuffer
        .iter()
        .zip(target.qual.iter())
        .enumerate()
        .for_each(|(i, (b, q))| {
            bases[tpos] = *b;
            quals[tpos] = *q;

            tpos += 1 + max_ins[i] as usize;
        });
}

fn get_features_for_window(
    overlaps: &mut [OverlapWindow],
    ovlps_cigar_map: &HashMap<u32, &Vec<u8>>,
    tid: u32,
    reads: &[HAECRecord],
    max_ins: &[u16],
    window_length: usize, // Full window length
    tbuffer: &[u8],
    qbuffer: &mut [u8],
) -> (Array2<u8>, Array2<u8>) {
    //Get features
    let length = max_ins.iter().map(|v| *v as usize).sum::<usize>() + max_ins.len();
    let width = 1 + overlaps.len();

    let mut bases = Array::from_elem((length, width), b'.');
    let mut quals = Array::from_elem((length, width), b'!');

    // First write the target
    write_target_for_window(
        &reads[tid as usize],
        &max_ins,
        bases.index_axis_mut(Axis(1), 0),
        quals.index_axis_mut(Axis(1), 0),
        window_length,
        tbuffer,
    );

    // Write top-k overlaps for the window
    overlaps.iter().enumerate().for_each(|(i, ow)| {
        let qid = ow.overlap.return_other_id(tid);
        get_features_for_ol_window(
            bases.index_axis_mut(Axis(1), i + 1),
            quals.index_axis_mut(Axis(1), i + 1),
            ow,
            ovlps_cigar_map.get(&qid).unwrap(),
            &reads[qid as usize],
            tid,
            &max_ins,
            qbuffer,
        )
    });


    (bases, quals)
}


fn overlap_window_filter(cigar: &[u8]) -> bool {
    let long_indel = CigarIter::new(cigar).any(|(op, _)| match op {
        CigarOp::Insertion(l) | CigarOp::Deletion(l) if l >= 30 => true,
        _ => false,
    });

    !long_indel
}

pub(crate) fn extract_features<'a, T: FeaturesOutput<'a>>(
    rid: u32,
    reads: &'a [HAECRecord],
    overlaps: Vec<Alignment>,
    // window_size: u32,
    module: &str,
    (tbuf, qbuf): (&mut [u8], &mut [u8]),
    feats_output: &mut T,
) {
    let read = &reads[rid as usize];
    reads[rid as usize].seq.get_sequence(tbuf);

    let window_size = read.seq.len();

    // Get overlaps for windows
    let n_windows = (read.seq.len() + window_size as usize - 1) / window_size as usize;
    let mut windows = vec![Vec::new(); n_windows];

    let mut ovlps_cigar_map = HashMap::default();
    for alignment in overlaps.iter() {
        let qid = alignment.overlap.return_other_id(rid);

        // let (tshift, qshift) = (0, 0);

        //Extract windows
        let is_target = alignment.overlap.tid == rid;
        extract_windows(
            &mut windows,
            &alignment.overlap,
            &alignment.cigar,
            // tshift,
            // qshift,
            is_target,
            // window_size,
        );

        ovlps_cigar_map.insert(qid, &alignment.cigar);

    }

    feats_output.init(rid, &read.id);
    for i in 0..n_windows {

        let win_len = read.seq.len();

        // Sort window to take TOP-K
        // Since there is just one window and for that window all information is in overlap, let not consider window specific variables.
        windows[i].sort_by_key(|ow| {
            let cigar = ovlps_cigar_map
                .get(&ow.overlap.return_other_id(rid))
                .unwrap();

            let tstart = ow.overlap.tstart as usize;
            let tend = ow.overlap.tend as usize;
            //reads[rid as usize].seq.get_subseq(tstart..tend, tbuf);

            let qid = ow.overlap.return_other_id(rid);
            let is_target = ow.overlap.tid == rid;
            // let (qstart, qend) = get_query_region(ow, rid);
            let qstart = ow.overlap.qstart as usize;
            let qend = ow.overlap.qend as usize;
            let qlen = (qend - qstart) as usize;
            match ow.overlap.strand {
                Strand::Forward => reads[qid as usize]
                    .seq
                    .get_subseq(qstart as usize..qend as usize, qbuf),
                Strand::Reverse => reads[qid as usize]
                    .seq
                    .get_rc_subseq(qstart as usize..qend as usize, qbuf),
            }
            let acc = 
                if is_target {
                    calculate_accuracy(ow, cigar, &tbuf[tstart..tend], &qbuf[..qlen])
                } else {
                    calculate_accuracy(ow, cigar, &qbuf[..qlen], &tbuf[tstart..tend])
                };
            OrderedFloat(-acc)
        });


        let max_ins = get_max_ins_for_window(
            &windows[i],
            &ovlps_cigar_map,
            rid,
            win_len,
        );

        let (full_bases, full_quals) = get_features_for_window(
            &mut windows[i],
            &ovlps_cigar_map,
            rid,
            reads,
            &max_ins,
            win_len,
            tbuf,
            qbuf,
        );

        let full_bases_t = full_bases.t().to_owned();
        let full_quals_t = full_quals.t().to_owned();
        let (bases_t, quals_t) = filter_rows_heuristic_three(&full_bases_t, &full_quals_t);
        let bases = bases_t.t().to_owned();
        let quals = quals_t.t().to_owned();

        // let bases = full_bases.to_owned();
        // let quals = full_quals.to_owned();


        let qids: Vec<&str> = windows[i]
            .iter()
            .map(|ow| {
                std::str::from_utf8(&reads[ow.overlap.return_other_id(rid) as usize].id).unwrap()
            })
            .collect();

        // let (supported, weakly_supported) = get_supported(&bases, &quals, module);
        let supported = get_supported(&full_bases, module);



        let qids_test: Vec<u32> = windows[i].iter().map(|ow| ow.overlap.qid).collect();

        feats_output.update(
            rid,
            i as u16,
            bases,
            quals,
            qids_test,
            supported,
            qids,
            n_windows as u16,
        );
    }

    feats_output.emit();
}



fn calculate_accuracy(window: &OverlapWindow, cigar: &[u8], tseq: &[u8], qseq: &[u8]) -> f32 {
    let (mut tpos, mut qpos) = (0, 0);
    let (mut m, mut s, mut i, mut d) = (0, 0, 0, 0);

    let cigar_iter = CigarIter::new(&cigar);
    for (op, range) in cigar_iter {
        let len = op.get_length() as usize;

        assert!(len > 0, "Operation length cannot be 0");

        // Not insertion -> consume tseq -> check bounds
        if !matches!(op, CigarOp::Insertion(_)) {
            assert!(
                tpos + len <= tseq.len(),
                "Length {} + {} is bigger than the tseq {}. {} {} {} {:?} {:?}",
                len,
                tpos,
                tseq.len(),
                window.cigar_start_offset,
                op.get_length(),
                std::str::from_utf8(cigar).unwrap(),
                range,
                window.cigar_start_idx..window.cigar_end_idx
            );
        }

        // Not deletion -> consume qseq -> check bounds
        if !matches!(op, CigarOp::Deletion(_)) {
            assert!(
                qpos + len <= qseq.len(),
                "Length {} + {} is bigger than the qseq {}. {} {} {} {:?} {:?}",
                len,
                qpos,
                qseq.len(),
                window.cigar_start_offset,
                op.get_length(),
                std::str::from_utf8(cigar).unwrap(),
                range,
                window.cigar_start_idx..window.cigar_end_idx
            );
        }

        match op {
            CigarOp::Match(_) => {
                for j in 0..len {
                    let tbase = tseq[tpos + j];
                    let qbase = qseq[qpos + j];

                    if tbase == qbase {
                        m += 1;
                    } else {
                        s += 1;
                    }
                }

                tpos += len;
                qpos += len;
            }
            CigarOp::Mismatch(_) => unreachable!(),
            CigarOp::Insertion(_) => {
                i += len;
                qpos += len;
            }
            CigarOp::Deletion(_) => {
                d += len;
                tpos += len;
            }
        }
    }

    // Alignment accuracy criteria giving more weight to match and mismatches
    // (m as f32) / ((5*s + m + i + d) as f32)

    (m as f32) / ((m + s + i + d) as f32)

}



fn get_supported<S>(bases: &ArrayBase<S, Ix2>, module: &str) -> Vec<SupportedPos>
where
    S: Data<Elem = u8>,
{

    let mut counter: HashMap<u8, u8> = HashMap::default();
    counter.insert(b'A', 0);
    counter.insert(b'C', 0);
    counter.insert(b'G', 0);
    counter.insert(b'T', 0);
    counter.insert(b'*', 0);

    let mut supporeted = Vec::new();

    let (mut tpos, mut ins) = (-1i16, 0);
    for col in bases.axis_iter(Axis(0)) {
        if col[0] == b'*' {
            ins += 1;
        } else {
            tpos += 1;
            ins = 0;
        }

        counter.iter_mut().for_each(|(_, c)| *c = 0);
        col.iter().for_each(|&b| {
            if b == b'.' {
                return;
            }

            // if b == b'*' || b == b'#' {
            //     return; // skip indels
            // }

            *counter.get_mut(&BASE_FORWARD[b as usize]).unwrap() += 1;
        });

        let n_supported = counter
            .iter()
            .fold(0u8, |acc, (_, &c)| if c >= 20 { acc + 1 } else { acc });
        if module != "consensus" && n_supported >= 2 {
            supporeted.push(SupportedPos::new(tpos as u16, ins));
        }
    }

    supporeted
}






fn output_features<P: AsRef<Path>>(
    path: P,
    window_id: u16,
    ids: &[&str],
    bases: Array2<u8>,
    quals: Array2<u8>,
    supported: impl IntoIterator<Item = SupportedPos>,
) -> Result<()> {
    let ids_path = path.as_ref().join(format!("{}.ids.txt", window_id));
    let ids_file = File::create(ids_path)?;
    let mut ids_writer = BufWriter::new(ids_file);
    for id in ids {
        writeln!(&mut ids_writer, "{}", id)?
    }

    let features_path = path.as_ref().join(format!("{}.features.npy", window_id));

    // Convert quals to u8 + stack feats
    let quals = quals.mapv(|q| q as u8);
    let features = stack![Axis(0), bases, quals];

    // Write feats
    let shape: Vec<_> = features.shape().iter().map(|&s| s as u64).collect();
    let mut writer = npyz::WriteOptions::new()
        .default_dtype()
        .shape(&shape)
        .writer(BufWriter::new(File::create(features_path)?))
        .begin_nd()?;
    writer.extend(features.iter())?;
    writer.finish()?;

    let supported_path = path.as_ref().join(format!("{}.supported.npy", window_id));
    let mut writer = npyz::WriteOptions::new()
        .default_dtype()
        .writer(BufWriter::new(File::create(supported_path)?))
        .begin_1d()?;
    writer.extend(supported)?;
    writer.finish()?;

    Ok(())
}

pub(crate) trait FeaturesOutput<'a> {
    fn init<'b>(&mut self, rid: u32, rname: &'b [u8])
    where
        'b: 'a;
    fn update(
        &mut self,
        rid: u32,
        wid: u16,
        bases: Array2<u8>,
        quals: Array2<u8>,
        qids: Vec<u32>,
        supported: Vec<SupportedPos>,
        ids: Vec<&str>,
        n_wids: u16,
    );
    fn emit(&mut self);
}

#[derive(Clone)]
pub(crate) struct FeatsGenOutput<'a, T>
where
    T: AsRef<Path> + Clone,
{
    base_path: T,
    rname: Option<&'a [u8]>,
    pbar_sender: Sender<PBarNotification>,
}

impl<T> FeatsGenOutput<'_, T>
where
    T: AsRef<Path> + Clone,
{
    pub(crate) fn new(path: T, pbar_sender: Sender<PBarNotification>) -> Self {
        Self {
            base_path: path,
            rname: None,
            pbar_sender: pbar_sender,
        }
    }
}

impl<'a, T> FeaturesOutput<'a> for FeatsGenOutput<'a, T>
where
    T: AsRef<Path> + Clone,
{
    fn init<'b>(&mut self, _rid: u32, rname: &'b [u8])
    where
        'b: 'a,
    {
        self.rname.replace(rname);
    }

    fn update(
        &mut self,
        _rid: u32,
        wid: u16,
        bases: Array2<u8>,
        quals: Array2<u8>,
        qids: Vec<u32>,
        supported: Vec<SupportedPos>,
        ids: Vec<&str>,
        _n_wids: u16,
    ) {
        let rid = std::str::from_utf8(self.rname.unwrap()).unwrap();
        let output_path = self.base_path.as_ref().join(rid);
        create_dir_all(&output_path).expect("Cannot create directory");

        output_features(&output_path, wid, &ids, bases, quals, supported.into_iter()).unwrap();
    }

    fn emit(&mut self) {
        self.pbar_sender.send(PBarNotification::Inc).unwrap();

        self.rname = None;
    }
}

pub(crate) struct CorrectOutput {
    sender: Sender<CorrectData>,
    features: Vec<WindowExample>,
    batch_size: usize,
}

impl CorrectOutput {
    pub(crate) fn new(sender: Sender<CorrectData>, batch_size: usize) -> Self {
        Self {
            sender,
            features: Vec::with_capacity(batch_size),
            batch_size: batch_size,
        }
    }
}

impl<'a> FeaturesOutput<'a> for CorrectOutput {
    fn init<'b>(&mut self, _rid: u32, _rname: &'b [u8])
    where
        'b: 'a,
    {
    }

    fn update(
        &mut self,
        rid: u32,
        wid: u16,
        bases: Array2<u8>,
        quals: Array2<u8>,
        qids: Vec<u32>,
        supported: Vec<SupportedPos>,
        ids: Vec<&str>,
        n_wids: u16,
    ) {
        self.features.push(WindowExample::new(
            rid,
            wid,
            ids.len() as u16,
            bases,
            quals,
            qids,
            supported,
            n_wids,
        ));

        if self.features.len() == self.batch_size {
            let data = prepare_examples(self.features.drain(..), self.batch_size);
            self.sender.send(data).unwrap();
        }
    }

    fn emit(&mut self) {
        let data = prepare_examples(self.features.drain(..), self.batch_size);
        self.sender.send(data).unwrap();
    }
}

#[derive(npyz::AutoSerialize, npyz::Serialize, PartialEq, Eq, Hash, Clone, Copy)]
#[derive(Debug)] 
pub(crate) struct SupportedPos {
    pub pos: u16,
    pub ins: u8,
}

impl SupportedPos {
    pub fn new(pos: u16, ins: u8) -> Self {
        SupportedPos { pos, ins }
    }
}
