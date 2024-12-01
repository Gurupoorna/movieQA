# Constructing small dataset for numerical query and retrieval

Data used : tmdb-5000-movies

The final dataset of query and document id pairs can be found in file 'full_annot_queries.json'.

### How to make your own
First construct template queries with numerical entities missing (use angular brackets for bracketing). Then list out the attribute to which a particular blank entity refers to and the relation instructed by the query for retrieval. With this done, you can run 'construct_full_annotations.py' over that file and save the results.