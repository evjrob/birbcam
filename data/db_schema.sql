CREATE TABLE results(
    utc_datetime text NOT NULL PRIMARY KEY,
    datetime text NOT NULL, 
    file_name text NOT NULL, 
    prediction text NOT NULL, 
    confidence real NOT NULL, 
    true_label text, 
    inaturalist_id INTEGER
);