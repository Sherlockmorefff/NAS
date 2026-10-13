"""Package only this completed A-only experiment; never train or score."""
import argparse
import gzip
import hashlib
import json
import tarfile
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    excluded = {'FILES.sha256', 'evidence.tar.gz', 'evidence.tar.gz.sha256'}
    files = []
    for path in sorted(root.rglob('*')):
        rel = path.relative_to(root)
        if any(part == '__pycache__' or part.startswith('._') for part in rel.parts):
            continue
        if str(rel) in excluded:
            continue
        if path.is_symlink():
            raise ValueError('Do not package symbolic links: ' + str(rel))
        if path.is_file():
            files.append(path)
    manifest = root / 'FILES.sha256'
    manifest.write_text(''.join(sha(p) + '  ' + str(p.relative_to(root)) + '\n' for p in files))
    files.append(manifest)
    archive = root / 'evidence.tar.gz'
    with archive.open('wb') as stream:
        with gzip.GzipFile(filename='', mode='wb', fileobj=stream, mtime=0, compresslevel=6) as compressed:
            with tarfile.open(fileobj=compressed, mode='w') as tar:
                for path in sorted(files):
                    info = tarfile.TarInfo(root.name + '/' + str(path.relative_to(root)))
                    info.size = path.stat().st_size
                    info.mode = 0o644
                    info.uid = info.gid = info.mtime = 0
                    info.uname = info.gname = ''
                    with path.open('rb') as content:
                        tar.addfile(info, content)
    digest = sha(archive)
    (root / 'evidence.tar.gz.sha256').write_text(digest + '  evidence.tar.gz\n')
    with tarfile.open(archive, mode='r:gz') as tar:
        members = tar.getmembers()
        assert len(members) == len(files)
        expected = {root.name + '/' + str(p.relative_to(root)): sha(p) for p in files}
        for member in members:
            assert member.isfile() and member.name in expected
            assert hashlib.sha256(tar.extractfile(member).read()).hexdigest() == expected[member.name]
    print(json.dumps({'pass': True, 'file_count_in_archive': len(files),
                      'files_manifest_sha256': sha(manifest), 'archive_sha256': digest,
                      'archive_bytes': archive.stat().st_size, 'archive_path': str(archive)}))


if __name__ == '__main__':
    main()
