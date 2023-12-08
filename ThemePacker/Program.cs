using System.Diagnostics;
using System.IO.Compression;

if (args.Length != 2) {
    Console.WriteLine("Invalid number of arguments");
    return;
}

if (!Directory.Exists(args[0])) {
    Console.WriteLine("Source directory does not exist");
    return;
}

try {
    if (!Directory.Exists(args[1]))
        Directory.CreateDirectory(args[1]);

    foreach (string dirPath in Directory.EnumerateDirectories(args[0])) {
        string dirName = Path.GetFileName(dirPath);
        string zipPath = Path.Combine(args[1], dirName + ".theme");

        if (File.Exists(zipPath))
            File.Delete(zipPath);

        ZipFile.CreateFromDirectory(dirPath, zipPath);
    }

    Console.WriteLine("Done Packing");
    Debug.WriteLine("Done Packing");
} catch (Exception e) {
    Console.WriteLine(e);
}

