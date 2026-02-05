import re
import os
import os.path
import pickle
import glob
import json
import argparse

class FileProcessing:
    
    def __init__(self, special_chars='-_', extension_list_path=None):
        self.special_chars = special_chars
        self.valid_extensions = self._load_valid_extensions(extension_list_path)
    
    def _load_valid_extensions(self, extension_list_path):
        if extension_list_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            extension_list_path = os.path.join(script_dir, 'extensionlist', 'Filename extension list')
        
        valid_extensions = set()
        
        try:
            with open(extension_list_path, 'r', encoding='utf-8') as f:
                for line in f:
                    ext = line.strip()
                    if ext and ext.startswith('.'):
                        valid_extensions.add(ext.lower())
                        valid_extensions.add(ext)
        except FileNotFoundError:
            pass
        return valid_extensions
    
    def _is_valid_extension(self, extension):
        if not extension:
            return False
        
        if not extension.startswith('.'):
            extension = '.' + extension
        
        return extension in self.valid_extensions or extension.lower() in self.valid_extensions
    
    def _remove_invalid_extension(self, filename):
        if '.' not in filename:
            return filename
        
        name, ext = os.path.splitext(filename)
        
        if len(ext) < 6:
            return filename
        
        if self._is_valid_extension(ext):
            return filename
        else:
            return name
    
    def _remove_numerics_simple(self, text):
        if not text:
            return ''
        
        if text.isdigit():
            return ''
        
        text = re.sub(r'\d+$', '', text)
        escaped_chars = re.escape(self.special_chars)
        text = re.sub(f'[{escaped_chars}]\\d+', '', text)
        text = re.sub(r'\d+', '', text)
        
        return text if text else ''
    
    def _normalize_path(self, path):
        normalized = ""
        i = 0
        while i < len(path):
            if path[i] == '\\':
                normalized += '/'
                while i+1 < len(path) and path[i+1] == '\\':
                    i += 1
            else:
                normalized += path[i]
            i += 1
        
        while '//' in normalized:
            normalized = normalized.replace('//', '/')
            
        return normalized
    
    def _is_file(self, path):
        norm_path = self._normalize_path(path)
        _, ext = os.path.splitext(norm_path)
        return bool(ext)
    
    def _remove_numerics(self, text):
        return self._remove_numerics_simple(text)
    
    def _process_filename_and_extension(self, filename):
        if '.' in filename:
            name, ext = filename.split('.', 1)
            processed_name = self._remove_numerics(name)
            return f"{processed_name}.{ext}"
        else:
            return self._remove_numerics(filename)
    
    def process_filenames(self, filenames):
        processed = []
        
        for filename in filenames:
            normalized = self._normalize_path(filename)
            
            if self._is_file(normalized):
                dirname = os.path.dirname(normalized)
                basename = os.path.basename(normalized)
                basename_no_invalid_ext = self._remove_invalid_extension(basename)
                
                if dirname:
                    dir_segments = dirname.split('/')
                    processed_dir_segments = []
                    for segment in dir_segments:
                        if segment:
                            cleaned = self._remove_numerics(segment)
                            if cleaned:
                                processed_dir_segments.append(cleaned)
                    processed_dir = '/'.join(processed_dir_segments)
                else:
                    processed_dir = ''
                
                processed_filename = self._process_filename_and_extension(basename_no_invalid_ext)
                
                if processed_dir:
                    processed_path = f"{processed_dir}/{processed_filename}"
                else:
                    processed_path = processed_filename
                
                processed.append(processed_path)
            else:
                segments = normalized.split('/')
                processed_segments = []
                for segment in segments:
                    if segment:
                        cleaned = self._remove_numerics(segment)
                        if cleaned:
                            processed_segments.append(cleaned)
                processed_path = '/'.join(processed_segments)
                processed.append(processed_path)
        
        return processed
    
    def create_clustering_json(self, original_paths, processed_paths, dataset_name):
        clustering = {}
        
        for original, processed in zip(original_paths, processed_paths):
            if processed not in clustering:
                clustering[processed] = []
            
            if original != processed:
                clustering[processed].append(original)
        
        for original, processed in zip(original_paths, processed_paths):
            if original == processed and original not in clustering:
                clustering[original] = []
        
        maps_dir = "maps"
        os.makedirs(maps_dir, exist_ok=True)
        
        output_filename = os.path.join(maps_dir, f"summarized-enames_{dataset_name.lower()}.json")
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(clustering, f, indent=2, ensure_ascii=False)
        
        return clustering
    
    def create_cluster_map_json(self, original_paths, processed_paths, dataset_name):
        cluster_map = {}
        
        for original, processed in zip(original_paths, processed_paths):
            cluster_map[original] = processed
        
        maps_dir = "maps"
        os.makedirs(maps_dir, exist_ok=True)
        
        output_filename = os.path.join(maps_dir, f"ename-cluster-map_{dataset_name.lower()}.json")
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(cluster_map, f, indent=2, ensure_ascii=False)
        
        return cluster_map
    
    def get_pkl_files(self, directory, dataset_name=None):
        if dataset_name:
            pkl_pattern = os.path.join(directory, f"enames_{dataset_name.lower()}.pkl")
        else:
            pkl_pattern = os.path.join(directory, "enames_*.pkl")
        return glob.glob(pkl_pattern)
    
    def process_pkl_files(self, directory, dataset_name=None):
        pkl_files = self.get_pkl_files(directory, dataset_name)
        
        if not pkl_files:
            return
        total_original_files = 0
        total_original_dirs = 0
        total_processed_files = 0
        total_processed_dirs = 0
        
        all_original_paths = []
        all_processed_paths = []
        all_dataset_data = []
        
        for pkl_file in pkl_files:
            filename = os.path.basename(pkl_file)
            dataset_name = filename.replace('.pkl', '').replace('enames_', '')
            
            try:
                with open(pkl_file, 'rb') as f:
                    original_filenames = pickle.load(f)
                
                if not isinstance(original_filenames, list):
                    original_filenames = list(original_filenames)
                
                files = []
                directories = []
                
                for path in original_filenames:
                    if self._is_file(path):
                        files.append(path)
                    else:
                        directories.append(path)
                
                original_file_count = len(files)
                original_dir_count = len(directories)
                original_total = original_file_count + original_dir_count
                
                total_original_files += original_file_count
                total_original_dirs += original_dir_count
                processed_files = []
                processed_dirs = []
                file_mappings = []
                dir_mappings = []
                
                if files:
                    processed_files_raw = self.process_filenames(files)
                    seen_files = set()
                    unique_processed_files = []
                    for original, processed in zip(files, processed_files_raw):
                        if processed not in seen_files and processed:
                            seen_files.add(processed)
                            unique_processed_files.append(processed)
                            file_mappings.append((original, processed))
                    processed_files = unique_processed_files
                
                if directories:
                    processed_dirs_raw = self.process_filenames(directories)
                    seen_dirs = set()
                    unique_processed_dirs = []
                    for original, processed in zip(directories, processed_dirs_raw):
                        if processed not in seen_dirs and processed:
                            seen_dirs.add(processed)
                            unique_processed_dirs.append(processed)
                            dir_mappings.append((original, processed))
                    processed_dirs = unique_processed_dirs
                
                processed_file_count = len(processed_files)
                processed_dir_count = len(processed_dirs)
                processed_total = processed_file_count + processed_dir_count
                
                total_processed_files += processed_file_count
                total_processed_dirs += processed_dir_count
                
                total_reduction = original_total - processed_total
                total_reduction_percent = (total_reduction / original_total * 100) if original_total > 0 else 0
                all_original_paths.extend(files + directories)
                all_processed_paths.extend(processed_files_raw + processed_dirs_raw)
                
                all_original_paths_dataset = files + directories
                all_processed_paths_dataset = processed_files_raw + processed_dirs_raw
                
                all_dataset_data.append({
                    'name': dataset_name,
                    'original': all_original_paths_dataset,
                    'processed': all_processed_paths_dataset
                })
                
                all_mappings = file_mappings + dir_mappings
            except Exception as e:
                pass
        for dataset_data in all_dataset_data:
            try:
                self.create_clustering_json(dataset_data['original'], dataset_data['processed'], dataset_data['name'])
                self.create_cluster_map_json(dataset_data['original'], dataset_data['processed'], dataset_data['name'])
            except Exception as e:
                pass


def test_mode():
    test_paths = [
        '/data/users/user123/profile',
        '/data/users/user456/profile',
        '/proc/123/stat',
        '/proc/1233/stat',
        '/path/to/abch/chroma',
        '/path/to/fefe/chroma',
        '/path/to/file12/chromium.v6512',
        '/path/to/file44/chromium.v6513',
        '/tmp/cache_abcd1234.tmp',
        '/tmp/cache_efgh5678.tmp',
        '/var/log/system.log.20230101',
        '/var/log/system.log.20230102',
        '/unique/file.txt',
        '/lib/_elliptic_curve.pyc',
        '/lib/_teletex_codec.pyc',
        '/sessions/session_abc123.log',
        '/sessions/session_def456.log',
        '/data/file_backup_001.dat',
        '/data/file_backup_002.dat',
        '/var/log/app.service.20230101',
        '/var/log/app.service.20230102',
        '/var/log/app.service.error.20230101',
        '/var/log/app.service.error.20230102',
        '/var/log/app.service.error.critical.20230101',
        '/var/log/app.service.error.critical.20230102'
    ] + [
        '/path/to/file/chromium.v6512',
        '/path/to/file/chromium.j6513',
        '/path/to/file/chromium.v7000',
        'path/to/chrome.exe',
        'path/to/svchost.exe',
        '/other/path/chromium.v8000',
        '/path/to/file/firefox.v1200',
        '/path/to/file/readme.txt',
        '/proc/123/stat',
        '/proc/1233/stat',
        '/proc/1133/stat',
        '/var/log/system.log.20230101',
        '/var/log/system.log.20230102',
        '/var/log/system.log.20230103',
        '/data/users/user123/profile',
        '/data/users/user456/profile',
        '/tmp/cache_abcd1234.tmp',
        '/tmp/cache_efgh5678.tmp',
        '/opt/software/v1.2.3/bin',
        '/opt/software/v1.2.4/bin',
        '/opt/software/v2.0.0/bin',
        '/backups/2023-01-01/db.sql',
        '/backups/2023-02-01/db.sql',
        '/projects/project-123/docs/readme.md',
        '/projects/project-456/docs/readme.md',
        '/sessions/a1b2c3d4-e5f6-7890-abcd-ef1234567890.sess',
        '/sessions/z9y8x7w6-v5u4-3210-wxyz-ab9876543210.sess',
        '/data/logs_2023_01_01-12_30.txt',
        '/data/logs_2023_01_02-13_45.txt',
        'C:\\Users\\user\\AppData\\Local\\Temp\\file1.tmp',
        'C:\\Users\\user\\AppData\\Local\\Temp\\file2.tmp',
        '\\\\Device\\\\HarddiskVolume2\\\\salt\\\\bin\\\\lib\\\\site-packages\\\\asn1crypto\\\\_elliptic_curve.pyc',
        '\\\\Device\\\\HarddiskVolume2\\\\salt\\\\bin\\\\lib\\\\site-packages\\\\asn1crypto\\\\_teletex_codec.pyc'
    ]
    processor = FileProcessing()
    for i, original_path in enumerate(test_paths, 1):
        normalized = processor._normalize_path(original_path)
        processed = processor.process_filenames([original_path])[0]
        is_file = processor._is_file(original_path)
    original_count = len(test_paths)
    processed_paths = processor.process_filenames(test_paths)
    seen = set()
    unique_processed = []
    for path in processed_paths:
        if path not in seen and path:
            seen.add(path)
            unique_processed.append(path)
    groups = {}
    for i, (original, processed) in enumerate(zip(test_paths, processed_paths)):
        if processed not in groups:
            groups[processed] = []
        groups[processed].append(original)
    sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)


def main():
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        test_mode()
        return
    
    parser = argparse.ArgumentParser(description='Process entity names from pickle files')
    parser.add_argument('--dataset', type=str, default=None, 
                       help='Dataset name to process (e.g., theia, fivedirections). If not specified, processes all enames_*.pkl files')
    
    args = parser.parse_args()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    processor = FileProcessing()
    
    processor.process_pkl_files(current_dir, args.dataset)


if __name__ == '__main__':
    main()
