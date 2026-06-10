from huggingface_hub import list_repo_tree
files = list(list_repo_tree('yangtao9009/PASD', path_in_repo='pasd_light'))
for f in files:
    name = f.rfilename if hasattr(f, 'rfilename') else f.path
    size = getattr(f, 'size', None)
    if size:
        print(f'{name}  ({size/1024/1024:.1f} MB)')
    else:
        print(name)
