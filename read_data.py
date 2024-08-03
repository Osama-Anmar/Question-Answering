def read_file(file_path, text_normalization):
    file = open(file_path, encoding='utf-8-sig').read().lower().split("\n")
    file =  [line.strip() for line in file if line.strip()] 
    file =  list(map(text_normalization, file))
    return file