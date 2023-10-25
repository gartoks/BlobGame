using System.Reflection;

namespace BlobGame.App;
internal static class Files {

    private static string SaveDirectory { get; } = "saves";
    //private static string SaveDirectory { get; } = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments), "BlobGame");
    private static string ResourceDirectory { get; } = "Resources";

    public static string GetSaveFilePath(string fileName) {
        if (!Directory.Exists(SaveDirectory))
            Directory.CreateDirectory(SaveDirectory);

        return Path.Combine(SaveDirectory, fileName);
    }

    public static string GetResourceFilePath(params string[] paths) {
        return Path.Combine(Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location), ResourceDirectory, Path.Combine(paths));
    }
}
