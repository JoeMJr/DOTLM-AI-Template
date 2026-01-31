# File to contain useful methods that I'll use to fill in any gaps in my code without bloating it


def count_string_in_large_file(filename, search_string):
    count = 0
    try:
        with open(filename, 'r') as file:
            for line in file:
                count += line.count(search_string)
        return count
    except FileNotFoundError:
        return f"Error: The file {filename} was not found."
    



