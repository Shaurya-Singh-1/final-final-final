The chunk_list helper is dropping the final partial chunk. For example, chunking [1, 2, 3, 4, 5] into size 2 should end with [5]. Fix the bug without changing the public API.
