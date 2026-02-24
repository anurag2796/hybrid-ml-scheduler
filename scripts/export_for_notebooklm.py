import os

def export_for_notebooklm():
    """
    Crawls the entire repository and concatenates all actual source code files
    into a single text file named notebookllm_source.txt.
    """
    # Exclude directories that are not source code or contain heavy binaries/dependencies
    IGNORED_DIRS = {
        '.git', '__pycache__', 'venv', '.venv', 'node_modules', 
        '.idea', '.vscode', 'archive', 'data', 'logs', 'dist', 'build', '.pytest_cache'
    }
    
    # Extensions to include
    ALLOWED_EXTS = {
        '.py', '.js', '.jsx', '.ts', '.tsx', '.md', '.json', '.yml', '.yaml', 
        '.sh', '.ini', '.txt', '.css', '.html'
    }
    
    # Base directory is the root of the project (parent of 'scripts')
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_file = os.path.join(base_dir, 'notebookllm_source.txt')
    
    print(f"Starting directory traversal from: {base_dir}")
    print(f"Outputting to: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for root, dirs, files in os.walk(base_dir):
            # Modify dirs in-place to skip ignored directories
            dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
            
            for file in files:
                # Check extension
                _, ext = os.path.splitext(file)
                if ext.lower() not in ALLOWED_EXTS:
                    continue
                
                # Exclude output file itself or large PDF documentation
                if file == 'notebookllm_source.txt' or file.endswith('.pdf'):
                    continue
                    
                file_path = os.path.join(root, file)
                # Ensure the path printed is relative to the base directory
                rel_path = os.path.relpath(file_path, base_dir)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        content = infile.read()
                        
                    # Write separator and file path inside
                    outfile.write(f"\n\n======= FILE: {rel_path} =======\n\n")
                    outfile.write(content)
                    
                except Exception as e:
                    print(f"Warning: Could not read {rel_path} - {e}")
                    
    print(f"Export complete. The file was saved at: {output_file}")

if __name__ == "__main__":
    export_for_notebooklm()
