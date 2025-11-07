// hale.rs

use ndarray::{Array1, Array2, Axis, ArrayView2};
use std::cmp::max;
use std::collections::HashMap;

// --- Constants and Mappings (Copied from inference.rs) ---


pub(crate) const BASES_MAP: [u8; 128] = [
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 9, 255, 255,
    255, 255, 255, 255, 4, 255, 255, 255, 10, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 1, 255, 255, 255, 2, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 3, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 5, 255, 6, 255, 255, 255, 7, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    8, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
];

// const BASES_UPPER: [u8; 10] = [b'A', b'C', b'G', b'T', b'*', b'A', b'C', b'G', b'T', b'*'];
const BASES_UPPER_COUNTER: [usize; 10] = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4];
const BASES_UPPER_COUNTER2: [u8; 10] = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4];
// const BASES_LOWER_COUNTER2: [u8; 10] = [5, 6, 7, 8, 9, 5, 6, 7, 8, 9];

const MIN_COV_TH: u32 = 18;
const MIN_K_TH: u32 = 6;





fn ilog2(x: u32) -> usize {
    assert!(x > 0); // log2 undefined for 0
    let y = u32::BITS - 1 - x.leading_zeros();
    y as usize
}

fn gray_encode(n: u32) -> u32 {
    n ^ (n >> 1)
}

fn gray_decode(mut g: u32) -> u32 {
    // Iteratively xor-shift without a loop, fixed width for u32
    g ^= g >> 16;
    g ^= g >> 8;
    g ^= g >> 4;
    g ^= g >> 2;
    g ^= g >> 1;
    g
}

fn next_gray_code(gray: u32) -> u32 {
    let idx = gray_decode(gray);
    gray_encode(idx + 1)
}


fn get_column_wise_row_indices(bases: &Array2<u8>, bases_with_col_indices: &mut Array2<u32>) {
    let n = bases.nrows();
    let m = bases.ncols();
    for col in 0..m {
        let mut row_count = 0;
        for row in 0..n {
            if bases[[row,col]] != 10 {
                bases_with_col_indices[[row_count,col]] = row as u32;
                row_count += 1;
            }
        }
    }
}


fn get_next_partition_cost(bitmask: u32, col: usize, bases: &Array2<u8>, bases_with_col_indices: &Array2<u32>, prev_bitmask: u32, base_counts: &mut [u32; 5]) -> usize {
    let n = bases.nrows();
    let set_bits = bitmask.count_ones() as usize;
    let bit_changed = bitmask ^ prev_bitmask;
    let idx = ilog2(bit_changed) + 1;
    let index = bases_with_col_indices[[idx,col]] as usize;

    // println!("base_counts: {:?}, set_bits: {}, col: {}, bitmask: {}, previous_bitmask: {}, idx: {}, index: {}", base_counts, set_bits, col, bitmask, prev_bitmask, idx, index);

    let base = bases[[index,col]];
    if bitmask&bit_changed == 0 {
        match base {
            x if x == BASES_MAP[b'A' as usize] || x == BASES_MAP[b'a' as usize] => base_counts[0] -= 1,
            x if x == BASES_MAP[b'C' as usize] || x == BASES_MAP[b'c' as usize] => base_counts[1] -= 1,
            x if x == BASES_MAP[b'G' as usize] || x == BASES_MAP[b'g' as usize] => base_counts[2] -= 1,
            x if x == BASES_MAP[b'T' as usize] || x == BASES_MAP[b't' as usize] => base_counts[3] -= 1,
            x if x == BASES_MAP[b'*' as usize] || x == BASES_MAP[b'#' as usize] => base_counts[4] -= 1,
            _ => (),
        }
    } else {
        match base {
            x if x == BASES_MAP[b'A' as usize] || x == BASES_MAP[b'a' as usize] => base_counts[0] += 1,
            x if x == BASES_MAP[b'C' as usize] || x == BASES_MAP[b'c' as usize] => base_counts[1] += 1,
            x if x == BASES_MAP[b'G' as usize] || x == BASES_MAP[b'g' as usize] => base_counts[2] += 1,
            x if x == BASES_MAP[b'T' as usize] || x == BASES_MAP[b't' as usize] => base_counts[3] += 1,
            x if x == BASES_MAP[b'*' as usize] || x == BASES_MAP[b'#' as usize] => base_counts[4] += 1,
            _ => (),
        }
    }
    let max_count = base_counts.iter().copied().max().unwrap_or(0) as usize;
    // println!("base_counts: {:?}, max_count: {}, set_bits: {}", base_counts, max_count, set_bits);
    set_bits + 1 - max_count

}


// function to get cost of a partition
fn get_first_partition_cost(bitmask: u32, col: usize, bases: &Array2<u8>, base_counts: &mut [u32; 5]) -> usize {
    let n = bases.nrows();
    let set_bits = bitmask.count_ones() as usize;
    // if set_bits < 6  {
    //     return u32::MAX as usize;
    // }
    // iterate over rows in col column of bases and count freq of b'A', b'C', b'T', b'G', b'*', b'a', b'c', b't', b'g', b'#'
    // let mut base_counts = [0u32; 5]; // [A, C, G, T, D]
    let mut row_count:u32 = 0;
    for row in 0..n {
        let base = bases[[row, col]];
        if base == BASES_MAP[b'.' as usize] {
            continue;
        }
        if (row > 0) && (bitmask & (1 << row_count)) == 0 {
            row_count += 1;
            continue; // skip rows that are not in the partition
        }
        match base {
            x if x == BASES_MAP[b'A' as usize] || x == BASES_MAP[b'a' as usize] => base_counts[0] += 1,
            x if x == BASES_MAP[b'C' as usize] || x == BASES_MAP[b'c' as usize] => base_counts[1] += 1,
            x if x == BASES_MAP[b'G' as usize] || x == BASES_MAP[b'g' as usize] => base_counts[2] += 1,
            x if x == BASES_MAP[b'T' as usize] || x == BASES_MAP[b't' as usize] => base_counts[3] += 1,
            x if x == BASES_MAP[b'*' as usize] || x == BASES_MAP[b'#' as usize] => base_counts[4] += 1,
            _ => (),
        }

        if row > 0 {
            row_count += 1;
        }
    }
    let max_count = base_counts.iter().copied().max().unwrap_or(0) as usize;
    // println!("base_counts: {:?}, max_count: {}, set_bits: {}, col: {}", base_counts, max_count, set_bits, col);
    set_bits + 1 - max_count
    
}


// function to get cost of a partition
fn get_partition_cost(bitmask: u32, col: usize, bases: &Array2<u8>) -> usize {
    let n = bases.nrows();
    let set_bits = bitmask.count_ones() as usize;
    if set_bits < MIN_K_TH as usize  {
        return u32::MAX as usize;
    }
    // iterate over rows in col column of bases and count freq of b'A', b'C', b'T', b'G', b'*', b'a', b'c', b't', b'g', b'#'
    let mut base_counts = [0u32; 5]; // [A, C, G, T, D]
    let mut row_count:u32 = 0;
    for row in 0..n {
        let base = bases[[row, col]];
        if base == BASES_MAP[b'.' as usize] {
            continue;
        }
        if (row > 0) && (bitmask & (1 << row_count)) == 0 {
            row_count += 1;
            continue; // skip rows that are not in the partition
        }
        match base {
            x if x == BASES_MAP[b'A' as usize] || x == BASES_MAP[b'a' as usize] => base_counts[0] += 1,
            x if x == BASES_MAP[b'C' as usize] || x == BASES_MAP[b'c' as usize] => base_counts[1] += 1,
            x if x == BASES_MAP[b'G' as usize] || x == BASES_MAP[b'g' as usize] => base_counts[2] += 1,
            x if x == BASES_MAP[b'T' as usize] || x == BASES_MAP[b't' as usize] => base_counts[3] += 1,
            x if x == BASES_MAP[b'*' as usize] || x == BASES_MAP[b'#' as usize] => base_counts[4] += 1,
            _ => (),
        }

        if row > 0 {
            row_count += 1;
        }
    }
    let max_count = base_counts.iter().copied().max().unwrap_or(0) as usize;
    // println!("base_counts: {:?}, max_count: {}, col: {}", base_counts, max_count, col);
    set_bits + 1 - max_count
    
}



fn get_common_and_diff_rows(prev_col: usize, col: usize, bases: &Array2<u8>) -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
    let n = bases.nrows();
    let mut common_curr_rows: Vec<u32> = Vec::new();
    let mut common_prev_rows: Vec<u32> = Vec::new();
    let mut diff_curr_rows: Vec<u32> = Vec::new();
    let mut diff_prev_rows: Vec<u32> = Vec::new();
    let mut prev_row_count = 0;
    let mut curr_row_count = 0;
    for row in 1..n {
        // check if both  bases[row,col] and bases[row,prev_col] have a non '-' character -> put index of that row in common
        // if bases[row,col] has a non '-' character and bases[row,prev_col] has a '-' character -> put index of that row in diff_curr
        // if bases[row,prev_col] has a non '-' character and bases[row,col] has a '-' character -> put index of that row in diff_prev
        if bases[[row, col]] != BASES_MAP[b'.' as usize] && bases[[row, prev_col]] != BASES_MAP[b'.' as usize] {
            common_prev_rows.push(prev_row_count);
            common_curr_rows.push(curr_row_count);
            prev_row_count += 1;
            curr_row_count += 1;
        } else if bases[[row, col]] != BASES_MAP[b'.' as usize] && bases[[row, prev_col]] == BASES_MAP[b'.' as usize] {
            diff_curr_rows.push(curr_row_count);
            curr_row_count += 1;
        } else if bases[[row, col]] == BASES_MAP[b'.' as usize] && bases[[row, prev_col]] != BASES_MAP[b'.' as usize] {
            diff_prev_rows.push(prev_row_count);
            prev_row_count += 1;
        }
    }
    (common_curr_rows, common_prev_rows, diff_curr_rows, diff_prev_rows)
}


fn backtrack(bases: &Array2<u8>, row_counts: &Vec<usize>, dp: &mut Array2<usize>, partition: &mut Vec<u8>) {
    let m = bases.ncols();
    let n = bases.nrows();

    let total_bits = row_counts[m-1] as u32;
    let mut min_cost: usize = u32::MAX as usize;
    let mut corr_mask: u32 = 0;
    for bitmask in 0..(1 << total_bits) {
        if dp[[m-1, bitmask as usize]] < min_cost {
            min_cost = dp[[m-1, bitmask as usize]];
            corr_mask = bitmask as u32;
        }
    }

    // println!("mincost_left: {}, bitmask: {:b}", min_cost, corr_mask);

    let mut corr_mask_cost = get_partition_cost(corr_mask, m-1, &bases);

    // println!("corr_mask: {:b}, corr_mask_cost: {}, min_cost: {}", corr_mask, corr_mask_cost, min_cost);

    // traverse col m-1, and if the ith non '-' char row and ith bit in corr_mask is set, set the partition[i] to 1
    let mut bit = 0;
    for row in 1..n {
        if bases[[row, m-1]] == BASES_MAP[b'.' as usize] {
            continue;
        }
        if corr_mask & (1 << bit) != 0 {
            partition[row] = 1;
        }
        bit += 1;
    }

    // trvaerse the columns from m-2 to 0, and if the ith non '-' char row and ith bit in corr_mask is set, set the partition[i] to 1
    for col in (0..m-1).rev() {
        let total_bits = row_counts[col] as u32;
        let mut bitmask = 0;


        // get common, diff_curr, diff_prev rows for this column
        let (curr_col_common_rows, prev_col_common_rows, curr_col_diff_rows, prev_col_diff_rows) = get_common_and_diff_rows(col+1, col, bases);

        // println!("common_curr: {:?}, common_prev: {:?}, diff_curr: {:?}, diff_prev: {:?}, column: {}",curr_col_common_rows, prev_col_common_rows, curr_col_diff_rows, prev_col_diff_rows, col);
        let prev_col_common_bits = prev_col_common_rows.len() as u32;
        let curr_col_common_bits = curr_col_common_rows.len() as u32;
        let prev_col_diff_bits = prev_col_diff_rows.len() as u32;
        let curr_col_diff_bits = curr_col_diff_rows.len() as u32;
 

        let mut curr_col_common_bitmask = 0 as u32;
        for i in 0..curr_col_common_bits as usize {
            if corr_mask & (1 << prev_col_common_rows[i]) != 0 {
                curr_col_common_bitmask |= 1 << curr_col_common_rows[i];
            }
        }
        // println!("common_curr bitmask: {:b}", curr_col_common_bitmask);

        let mut temp = curr_col_common_bitmask;
        bitmask = 0 as u32;
        while bitmask < (1 << curr_col_diff_bits) {
            // of ith bit in bitmask is set, then set the curr_col_diff_rows[i]th bit in temp
            for i in 0..curr_col_diff_bits as usize {
                if (bitmask & (1 << i) as u32) != 0 {
                    temp |= 1 << curr_col_diff_rows[i];
                }
            }
            let cost_c = dp[[col, temp as usize]];
            // println!("temp: {:b}, cost_c {}", temp, cost_c);
            if cost_c + corr_mask_cost == min_cost {
                corr_mask = temp;
                min_cost = cost_c;
                corr_mask_cost = get_partition_cost(corr_mask, col, &bases);
                break;
            }

            temp = curr_col_common_bitmask;
            // increment the bitmask to the next one with same number of set bits
            bitmask += 1;
        }

        // println!("corr_mask: {:b}, corr_mask_cost: {}, min_cost: {}", corr_mask, corr_mask_cost, min_cost);
        bit = 0;
        for row in 1..n {
            if bases[[row, col]] == BASES_MAP[b'.' as usize] {
                continue;
            }
            if corr_mask & (1 << bit) != 0 {
                partition[row] = 1;
            }
            bit += 1;
        }
    }

    // println!("partition: {:?}", partition);
}


fn get_dp_for_hale_update(bases: &Array2<u8>, row_counts: &Vec<usize>, dp: &mut Array2<usize>, bases_with_col_indices: &Array2<u32>) {
    let m = bases.ncols();

    // for column 0, we get all those bitmasks that set exactly k bits out of max_count bits
    // For all these columns, we get their cost and set cost of that bitmask in dp[0][bitmask]
    let mut test_mn = u32::MAX as usize;
    let total_bits = row_counts[0] as u32;
    let mut bitmask = 0 as u32;
    let mut bitmask_binary = 0 as u32;
    let mut base_counts = [0u32; 5];
    let mut previous_bitmask = 0 as u32;
    while bitmask_binary < (1 << total_bits) {
        // same as bitmask being 0, which means no bits are set and hence cost will be u32::MAX
        let cost = if previous_bitmask == bitmask  {
            get_first_partition_cost(bitmask, 0, &bases, &mut base_counts)
        } else {
            get_next_partition_cost(bitmask, 0, &bases, &bases_with_col_indices, previous_bitmask, &mut base_counts)
        };
        //naive
        // let cost = get_partition_cost(bitmask_binary,0,&bases);

        // println!("bitmask: {:b}, cost: {}", bitmask, cost);

        let bitmask_set_bits = bitmask.count_ones() as usize;
        if bitmask_set_bits >= MIN_K_TH as usize {
            dp[[0, bitmask as usize]] = cost;
            test_mn = test_mn.min(cost);
        }
        previous_bitmask = bitmask;
        bitmask = next_gray_code(bitmask);
        bitmask_binary += 1;

    }

    // println!("checkpoint 1");
    // println!("test_mn: {}", test_mn);
    // Iterate over all columns from 1 to m-1
    // For each column, we iterate over all bitmasks of size 2^(row_counts[col])
    for col in 1..m {
        test_mn = u32::MAX as usize;
        bitmask = 0;
        bitmask_binary = 0;
        previous_bitmask = 0;

        let mut base_counts_outer = [0u32; 5];

        // get common, diff_curr, diff_prev rows for this column
        let (curr_col_common_rows, prev_col_common_rows, curr_col_diff_rows, prev_col_diff_rows) = get_common_and_diff_rows(col-1, col, bases);
        // println!("common_curr: {:?}, common_prev: {:?}, diff_curr: {:?}, diff_prev: {:?}, column: {}",curr_col_common_rows, prev_col_common_rows, curr_col_diff_rows, prev_col_diff_rows, col);
        let curr_col_common_bits = curr_col_common_rows.len() as usize;
        let prev_col_diff_bits = prev_col_diff_rows.len() as usize;
        let curr_col_diff_bits = curr_col_diff_rows.len() as usize;

        let mut prev_col_outer_bitmask = 0 as u32;
        let mut curr_col_outer_bitmask = 0 as u32;
        let mut prev_iter_curr_col_outer_bitmask = 0 as u32;

        while bitmask_binary < (1 << curr_col_common_bits) {
            // if previous_bitmask == bitmask it means bitmask = 0, which means there is nothing to set, both prev_col_outer_bitmask and curr_col_outer_bitmask will be 0
            if previous_bitmask != bitmask {
                let bit_changed = bitmask ^ previous_bitmask;
                let index_changed = ilog2(bit_changed);
                if bitmask & bit_changed == 0 {
                    prev_col_outer_bitmask ^=  1 << prev_col_common_rows[index_changed];
                    curr_col_outer_bitmask ^= 1 << curr_col_common_rows[index_changed];
                } else {
                    prev_col_outer_bitmask |= 1 << prev_col_common_rows[index_changed];
                    curr_col_outer_bitmask |= 1 << curr_col_common_rows[index_changed];
                }
            }

            //naive
            // for i in 0..curr_col_common_bits {
            //     if bitmask_binary & (1<<i) != 0 {
            //         prev_col_outer_bitmask |= 1 << prev_col_common_rows[i];
            //         curr_col_outer_bitmask |= 1 << curr_col_common_rows[i];
            //     }
            // }

            // now we have prev_col_outer_bitmask, we can iterate over all bitmasks in prev col that are compatible with prev_col_outer_bitmask mask and store the cost of min bitmask
            let mut prev_col_inner_bitmask = prev_col_outer_bitmask;
            let mut min_cost = u32::MAX as usize;
            let mut temp_bitmask = 0;
            let mut temp_bitmask_binary = 0;
            let mut previous_temp_bitmask = 0;
            while temp_bitmask_binary < (1 << prev_col_diff_bits) {
                // if previous_temp_bitmask == temp_bitmask, it means temp_bitmask = 0, which means the cost of inner bitmask is same as outer bitmask of the previous column
                if previous_temp_bitmask != temp_bitmask {
                    let bit_changed = temp_bitmask ^ previous_temp_bitmask;
                    let index_changed = ilog2(bit_changed);
                    if temp_bitmask & bit_changed == 0 {
                        prev_col_inner_bitmask ^= 1 << prev_col_diff_rows[index_changed];
                    } else {
                        prev_col_inner_bitmask |= 1 << prev_col_diff_rows[index_changed];
                    }
                }

                // naive
                // for i in 0..prev_col_diff_bits {
                //     if temp_bitmask_binary & (1<<i) != 0 {
                //         prev_col_inner_bitmask |= 1 << prev_col_diff_rows[i];
                //     }
                // }

                let temp_bitmask_cost = dp[[col-1, prev_col_inner_bitmask as usize]];
                if temp_bitmask_cost < min_cost {
                    min_cost = temp_bitmask_cost;
                }

                // naive
                // prev_col_inner_bitmask = prev_col_outer_bitmask;       

                previous_temp_bitmask = temp_bitmask;
                temp_bitmask = next_gray_code(temp_bitmask);
                temp_bitmask_binary += 1;
            }

            // if(min_cost == u32::MAX as usize) {
            //     // if min_cost is u32::MAX, it means there is no valid partition for this column, so we can skip this bitmask
            //     prev_iter_curr_col_outer_bitmask = curr_col_outer_bitmask;
            //     previous_bitmask = bitmask;
            //     bitmask = next_gray_code(bitmask);
            //     bitmask_binary += 1;
            //     continue;
            // }

            // Since we also have curr_col_outer_bitmask, we can iterate over all bitmasks in curr col that are compatible with curr_col_outer_bitmask mask and calculate the cost of each bitmask
            // and store the cost of each bitmask plus the min cost of prev col bitmask in dp table
            let mut curr_col_inner_bitmask = curr_col_outer_bitmask;
            temp_bitmask = 0 as u32;
            previous_temp_bitmask = 0 as u32;
            temp_bitmask_binary = 0;
            let mut temp_bitmask_cost = 0 as usize;
            let mut base_counts_inner = base_counts_outer.clone();

            while temp_bitmask_binary < (1 << curr_col_diff_bits) {
                // if previous_temp_bitmask == temp_bitmask, it means temp_bitmask = 0, which means the cost of inner bitmask is same as outer bitmask of the current column
                if previous_temp_bitmask == temp_bitmask {
                    // this is also the cost of curr_Col_outer_bitmask ans will be usefull in next iteration
                    // if bitmask is zero, there is no previous iteration to this, hence we will caluclate the cost of this bitmask naively, which we ofcourse know is u32::MAX, but we also have to update the base_counts
                    temp_bitmask_cost = if bitmask == 0 {
                        get_first_partition_cost(curr_col_inner_bitmask, col, &bases, &mut base_counts_inner)
                    } else {
                        get_next_partition_cost(curr_col_inner_bitmask, col, &bases, &bases_with_col_indices, prev_iter_curr_col_outer_bitmask, &mut base_counts_inner)
                    };
                    // since this is very first iteration where curr_col_inner_bitmask = curr_col_outer_bitmask, we keep the basecounts after this iteration 
                    // stored in base_counts_outer to be used at start of this loop before next iteration
                    base_counts_outer = base_counts_inner.clone();
                } else {
                    let bit_changed = temp_bitmask ^ previous_temp_bitmask;
                    let index_changed = ilog2(bit_changed);
                    let curr_col_inner_bitmask_before_update = curr_col_inner_bitmask;
                    if temp_bitmask & bit_changed == 0 {
                        curr_col_inner_bitmask ^= 1 << curr_col_diff_rows[index_changed];
                    } else {
                        curr_col_inner_bitmask |= 1 << curr_col_diff_rows[index_changed];
                    }
                    temp_bitmask_cost = get_next_partition_cost(curr_col_inner_bitmask, col, &bases, bases_with_col_indices, curr_col_inner_bitmask_before_update, &mut base_counts_inner);
                    
                }

                // naive
                // for i in 0..curr_col_diff_bits {
                //     if temp_bitmask_binary & (1<<i) != 0 {
                //         curr_col_inner_bitmask |= 1 << curr_col_diff_rows[i];
                //     }
                // }
                // temp_bitmask_cost = get_partition_cost(curr_col_inner_bitmask, col, &bases);

                let set_bits_in_curr_col_inner_bitmask = curr_col_inner_bitmask.count_ones() as usize;
                dp[[col, curr_col_inner_bitmask as usize]] = if min_cost < u32::MAX as usize && set_bits_in_curr_col_inner_bitmask >= MIN_K_TH as usize {
                    min_cost + temp_bitmask_cost
                } else {
                    u32::MAX as usize
                };
                test_mn = test_mn.min(dp[[col, curr_col_inner_bitmask as usize]]);

                // naive
                // curr_col_inner_bitmask = curr_col_outer_bitmask;

                previous_temp_bitmask = temp_bitmask;
                temp_bitmask = next_gray_code(temp_bitmask);
                temp_bitmask_binary += 1;

            }

            prev_iter_curr_col_outer_bitmask = curr_col_outer_bitmask;
            previous_bitmask = bitmask;
            bitmask = next_gray_code(bitmask);
            bitmask_binary += 1;
        }
        // println!("test_mn in col: {}, {}", test_mn, col);

    }

    // println!("dp {:#?}", dp);

}


fn compute_corrected_seq(bases: &Array2<u8>, partition: &Vec<u8>, corrections: &mut Vec<u8>) {
    let m = bases.ncols();
    let n = bases.nrows();
    for i in 0..m {
        let mut counts: Vec<u8> = vec![0; 5]; // Assuming 5 possible bases (A, T, C, G, '*')
    
        for j in 0..n {
            if partition[j] != 0 {
                let base = bases[[j, i]];
                if base != BASES_MAP[b'.' as usize] {
                    counts[BASES_UPPER_COUNTER[base as usize]] += 1;
                }
            }
        }

        if let Some(max_count) = counts.iter().max() {
            let mut best_index = None;
            let mut tie = false;
        
            for (idx, &count) in counts.iter().enumerate() {
                if count == *max_count {
                    if best_index.is_some() {
                        tie = true;
                    }
                    best_index = Some(idx);
                }
            }
        
            if let Some(best_idx) = best_index {
                if tie {
                    let target_base_idx = BASES_UPPER_COUNTER[bases[[0, i]] as usize];
                    if counts[target_base_idx] == *max_count {
                        best_index = Some(target_base_idx);
                    }
                }
                corrections[i] = BASES_UPPER_COUNTER2[best_index.unwrap()];
                // corrections[i] = BASES_LOWER_COUNTER2[best_index.unwrap()];
            }

        }
        
    }

}


/// Finds the optimal correction sequence using the best partition
pub fn hale_updated(bases: &Array2<u8>) -> Vec<u8> {
    let n = bases.nrows();
    let m = bases.ncols();
    // println!("n: {}, m: {}", n, m);
    if(n < MIN_COV_TH as usize || m == 0) {
        let row0 = bases.row(0);
        return row0.to_vec();
    }

    // a preprocess_step to actually get the row numbers for each clumn where ther is a non '-' character
    let mut bases_with_col_indices: Array2<u32> = Array2::from_elem(bases.raw_dim(), u32::MAX);
    get_column_wise_row_indices(&bases, &mut bases_with_col_indices);

    // println!("bases_with_col_indices: {:#?}", bases_with_col_indices);


    // create a vector containinf the count of non '-' characters in each column
    let mut row_counts: Vec<usize> = bases
        .axis_iter(Axis(1)) // iterate over columns
        .map(|col| col.iter().filter(|&&b| b != BASES_MAP[b'.' as usize]).count())
        .collect();
    
    // subtract 1 from each count to account for the first row
    row_counts = row_counts.iter()
        .map(|&count| if count > 0 { count - 1 } else { 0 }) // subtract 1 from each count
        .collect();

    let min_count = *row_counts.iter().min().unwrap_or(&0) as u32;
    let max_count = *row_counts.iter().max().unwrap_or(&0) as u32;
    // println!("min_count: {}, max_count: {}", min_count, max_count);  
    if min_count < MIN_COV_TH {
        // println!("min_count < th, returning pih");
        // panic!("min_count < th, returning pih, min_count: {}, total_bits: {}, bases.ncols(): {}", min_count, total_bits, bases.ncols());
        // later update it to consesnus or a better strategy
        // return two_approx_hale(bases);
        let row0 = bases.row(0);
        return row0.to_vec();
    }

    // init a matrix called dp of size m x 2^(max_count) with all values set to infinity
    let mut dp = Array2::<usize>::from_elem((m, 1 << (max_count) as usize), u32::MAX as usize);

    // solve hale update to get the dp table
    get_dp_for_hale_update(&bases, &row_counts, &mut dp, &bases_with_col_indices);
    
    // println!("time for backtrack");


    // backtrack
    let mut partition = vec![0; n];
    partition[0] = 1;
    backtrack(&bases, &row_counts, &mut dp, &mut partition);
    // partition should be set by now!
    // println!("partition: {:?}", partition);


    // Compute the corrected sequence
    let mut corrections: Vec<u8> = vec![4; m]; // Default to '*' if no match
    compute_corrected_seq(&bases, &partition, &mut corrections);
    // println!{"bases: {:?}", bases};
    // println!{"corrections: {:?}", corrections};

    corrections

}




