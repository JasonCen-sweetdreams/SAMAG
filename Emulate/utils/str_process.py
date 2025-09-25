def remove_before_first_space(s:str):
    parts = s.split(' ', 1)
    if len(parts) > 1:
        return parts[1].replace("\"","")
    else:
        return s.replace("\"","")