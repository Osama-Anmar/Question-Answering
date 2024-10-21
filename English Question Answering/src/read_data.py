def read_file(file_path):
    file = open(file_path, encoding='utf-8-sig').read().lower().split("\n")
    file =  [line.strip() for line in file if line.strip()] 
    return file