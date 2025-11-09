use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::error::Error;
use serde::Deserialize;
use std::sync::{Arc, OnceLock};

#[derive(Debug, Deserialize)]
struct Record {
    read_id: String,
    haplotype: String,
    position: u64,
    correct_base: char,
}

pub type HetSitesMap = HashMap<u32, Vec<(u64, char)>>;

/// Global holder for the map (set once at runtime)
pub static HET_SITES_MAP: OnceLock<Arc<HetSitesMap>> = OnceLock::new();

/// Initialize the het sites map using name_to_id
pub fn init_het_sites_map<P: AsRef<Path>>(
    path: P,
    name_to_id: &HashMap<String, u32>,
) -> Arc<HetSitesMap> {
    let map = load_het_sites_map(path, name_to_id)
        .unwrap_or_else(|e| panic!("Failed to load het sites map: {}", e));
    let arc_map = Arc::new(map);

    // Store globally if not already set
    if let Err(_) = HET_SITES_MAP.set(arc_map.clone()) {
        eprintln!("Warning: HET_SITES_MAP already initialized, skipping re-init");
    }

    arc_map
}

/// Internal loader from CSV → HetSitesMap
// het_sites_map.rs
// ...

/// Internal loader from CSV → HetSitesMap
fn load_het_sites_map<P: AsRef<Path>>(
    path: P,
    name_to_id: &HashMap<String, u32>,
) -> Result<HetSitesMap, Box<dyn Error>> {
    let file = File::open(path)?;
    // Use csv::Reader::from_reader(file) for standard reading
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true) // Assuming your CSV has a header line
        .from_reader(file);

    let mut map: HetSitesMap = HashMap::new();

    for result in rdr.deserialize() {
        let mut record: Record = result?;
        
        // --- FIX: Trim whitespace from the CSV read_id ---
        let trimmed_read_id = record.read_id.trim(); 
        
        if let Some(&id) = name_to_id.get(trimmed_read_id) {
            map.entry(id)
                .or_default()
                .push((record.position, record.correct_base));
        } else {
            // println!(
            //     "Warning: read_id '{}' not found in name_to_id mapping, skipping.",
            //     record.read_id
            // );
        }
    }

    Ok(map)
}