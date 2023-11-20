
using System.IO.Compression;

if (args.Length != 2)
    throw new Exception("Invalid number of arguments");

if (!Directory.Exists(args[0]))
    throw new Exception("Source directory does not exist");

if (!Directory.Exists(args[1]))
    Directory.CreateDirectory(args[1]);

foreach (string dirPath in Directory.EnumerateDirectories(args[0])) {
    string dirName = Path.GetFileName(dirPath);
    string zipPath = Path.Combine(args[1], dirName + ".theme");

    if (File.Exists(zipPath))
        File.Delete(zipPath);

    ZipFile.CreateFromDirectory(dirPath, zipPath);
}
