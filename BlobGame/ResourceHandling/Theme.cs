using BlobGame.Audio;
using OpenTK.Mathematics;
using SimpleGL.Graphics;
using SimpleGL.Graphics.Textures;
using SimpleGL.Util;
using SixLabors.Fonts;
using StbImageSharp;
using System.Globalization;
using System.IO.Compression;
using System.Text.Json;

namespace BlobGame.ResourceHandling;
/// <summary>
/// Class for one set of game resources. Doesn't cache anything.
/// </summary>
internal sealed class Theme : IDisposable, IEquatable<Theme?> {
    public string Name { get; }

    /// <summary>
    /// The file path to the asset file.
    /// </summary>
    private string ThemeFilePath { get; }

    /// <summary>
    /// A mapping of color name to colors.
    /// </summary>
    private Dictionary<string, Color4> Colors { get; }

    /// <summary>
    /// The zip archive containing all assets.
    /// </summary>
    private ZipArchive? ThemeArchive { get; set; }

    /// <summary>
    /// Flag indicating whether the theme was loaded.
    /// </summary>
    private bool WasLoaded { get; set; }

    private bool disposedValue;

    /// <summary>
    /// Constructor to load a theme from disk.
    /// </summary>
    internal Theme(string themefilePath) {
        Name = Path.GetFileNameWithoutExtension(themefilePath);

        ThemeFilePath = themefilePath;
        Colors = new Dictionary<string, Color4>();

        WasLoaded = false;
    }

    internal void Load() {
        if (WasLoaded)
            throw new InvalidOperationException("Theme was already loaded.");

        Log.WriteLine($"Loading theme {Name}");
        MemoryStream ms = new MemoryStream();
        using FileStream fs = new FileStream(ThemeFilePath, FileMode.Open);

        fs.CopyTo(ms);
        ms.Position = 0;

        ThemeArchive = new ZipArchive(ms, ZipArchiveMode.Read);

        ZipArchiveEntry? colorEntry = ThemeArchive.GetEntry("colors.json");
        if (colorEntry == null) {
            Log.WriteLine($"Theme {ThemeFilePath} doesn't contain colors.");
            return;
        }

        StreamReader colorStreamReader = new StreamReader(colorEntry.Open());
        Dictionary<string, int[]>? colors = JsonSerializer.Deserialize<Dictionary<string, int[]>>(colorStreamReader.ReadToEnd());
        if (colors == null) {
            Log.WriteLine($"colors.json in theme {ThemeFilePath} has a wrong format.");
            return;
        }

        foreach (KeyValuePair<string, int[]> entry in colors) {
            Colors[entry.Key] = new Color4((byte)entry.Value[0], (byte)entry.Value[1], (byte)entry.Value[2], (byte)entry.Value[3]);
        }

        WasLoaded = true;
    }

    internal void Unload() {

    }

    /// <summary>
    /// Tries to get a color to the given key.
    /// </summary>
    /// <param name="key"></param>
    /// <exception cref="InvalidOperationException">Thrown if the theme was not loaded.</exception>
    internal Color4? GetColor(string key) {
        if (!WasLoaded)
            throw new InvalidOperationException("Theme was not loaded.");

        if (!Colors.ContainsKey(key)) {
            return null;
        }

        return Colors[key];
    }

    /// <summary>
    /// Tries to load a font from the zip archive.
    /// </summary>
    /// <param name="key"></param>
    /// <exception cref="InvalidOperationException">Thrown if the theme was not loaded.</exception>
    public FontFamilyData? LoadFont(string key) {
        if (!WasLoaded)
            throw new InvalidOperationException("Theme was not loaded.");

        bool isTTF = true;

        string path = $"Fonts/{key}.ttf";
        ZipArchiveEntry? zippedFont = ThemeArchive!.GetEntry(path);

        if (zippedFont == null) {
            isTTF = false;
            path = $"Fonts/{key}.otf";
            zippedFont = ThemeArchive!.GetEntry(path);
        }

        FontFamily fontFamily;
        if (zippedFont != null) {
            Stream fontStream = zippedFont.Open();
            MemoryStream ms = new MemoryStream();
            fontStream.CopyTo(ms);
            fontStream.Dispose();
            ms.Position = 0;

            FontCollection fontCollection = new FontCollection();
            fontFamily = fontCollection.Add(ms);
            ms.Dispose();

        } else if (SystemFonts.Collection.TryGet(key, CultureInfo.InvariantCulture, out fontFamily)) {
            // Do nothing
        } else {
            Log.WriteLine($"Font {key} doesn't exist in this theme");
            return null;
        }

        return new FontFamilyData(fontFamily, isTTF);
    }

    /// <summary>
    /// Tries to load a texture from the zip archive.
    /// </summary>
    /// <param name="key"></param>
    /// <exception cref="InvalidOperationException">Thrown if the theme was not loaded.</exception>
    public Texture2D? LoadTexture(string key) {
        if (!WasLoaded)
            throw new InvalidOperationException("Theme was not loaded.");

        string path = $"Textures/{key}.png";
        ZipArchiveEntry? zippedTexture = ThemeArchive!.GetEntry(path);

        if (zippedTexture == null) {
            Log.WriteLine($"Texture {key} doesn't exist in this theme");
            return null;
        }

        Stream textureStream = zippedTexture.Open();
        MemoryStream ms = new MemoryStream();
        textureStream.CopyTo(ms);
        textureStream.Dispose();
        ms.Position = 0;

        ImageResult image = ImageResult.FromStream(ms, ColorComponents.RedGreenBlueAlpha);
        ms.Dispose();

        Texture2D texture = GraphicsHelper.CreateTexture(image);

        // TODO add check if texture was loaded correctly
        /*if (texture.id == 0) {
            Log.WriteLine($"Failed to load texture {key} from {path}");
            return null;
        }*/
        return texture;
    }

    /// <summary>
    /// Tries to load a sound from the zip archive.
    /// </summary>
    /// <param name="key"></param>
    /// <exception cref="InvalidOperationException">Thrown if the theme was not loaded.</exception>
    public Sound? LoadSound(string key) {
        if (!WasLoaded)
            throw new InvalidOperationException("Theme was not loaded.");

        string path = $"Sounds/{key}.wav";
        ZipArchiveEntry? zippedSound = ThemeArchive!.GetEntry(path);

        if (zippedSound == null) {
            Log.WriteLine($"Sound {key} doesn't exist in this theme");
            return null;
        }

        Stream soundStream = zippedSound.Open();
        MemoryStream ms = new MemoryStream();
        soundStream.CopyTo(ms);
        soundStream.Dispose();
        ms.Position = 0;

        Sound sound = Sound.Create(ms);
        ms.Dispose();

        return sound;
    }

    /// <summary>
    /// Tries to load a music from the zip archive.
    /// </summary>
    /// <param name="key"></param>
    /// <exception cref="InvalidOperationException">Thrown if the theme was not loaded.</exception>
    public Music? LoadMusic(string key) {
        if (!WasLoaded)
            throw new InvalidOperationException("Theme was not loaded.");

        string path = $"Music/{key}.mp3";
        ZipArchiveEntry? zippedSound = ThemeArchive!.GetEntry(path);

        if (zippedSound == null) {
            Log.WriteLine($"Music {key} doesn't exist in this theme");
            return null;
        }

        Stream musicStream = zippedSound.Open();
        MemoryStream ms = new MemoryStream();
        musicStream.CopyTo(ms);
        musicStream.Dispose();
        ms.Position = 0;

        Music music = Music.Create(ms);
        ms.Dispose();

        return music;
    }

    /// <summary>
    /// Tries to get a text to the given key.
    /// </summary>
    /// <param name="key"></param>
    /// <returns></returns>
    /// <exception cref="InvalidOperationException">Thrown if the theme was not loaded.</exception>
    internal IReadOnlyDictionary<string, string>? LoadText(string key) {
        if (!WasLoaded)
            throw new InvalidOperationException("Theme was not loaded.");

        string path = $"Texts/{key}.json";
        ZipArchiveEntry? zippedText = ThemeArchive!.GetEntry(path);

        if (zippedText == null) {
            Log.WriteLine($"Text {key} doesn't exist in this theme");
            return null;
        }

        using Stream textStream = zippedText.Open();
        Dictionary<string, string>? dict = JsonSerializer.Deserialize<Dictionary<string, string>>(textStream);

        return dict;
    }

    /// <summary>
    /// Tries to load a NPatchTexture from the zip archive.
    /// </summary>
    /// <param name="key"></param>
    /// <exception cref="InvalidOperationException">Thrown if the theme was not loaded.</exception>
    public NPatchTexture? LoadNPatchTexture(string key) {
        if (!WasLoaded)
            throw new InvalidOperationException("Theme was not loaded.");

        Texture2D? texture = LoadTexture(key);

        if (texture == null)
            return null;

        string path = $"Textures/NPatchData/{key}.json";
        ZipArchiveEntry? zippedText = ThemeArchive!.GetEntry(path);

        if (zippedText == null) {
            Log.WriteLine($"NPatchData {key} doesn't exist in this theme");
            return null;
        }

        using Stream textStream = zippedText.Open();
        Dictionary<string, int>? dict = JsonSerializer.Deserialize<Dictionary<string, int>>(textStream);

        if (dict == null)
            return null;

        return new NPatchTexture(texture, dict["left"], dict["right"], dict["top"], dict["bottom"]);
    }

    private void Dispose(bool disposing) {
        if (!disposedValue) {
            if (disposing) {
                ThemeArchive?.Dispose();
            }

            // TODO: free unmanaged resources (unmanaged objects) and override finalizer
            // TODO: set large fields to null
            disposedValue = true;
        }
    }

    // // TODO: override finalizer only if 'Dispose(bool disposing)' has code to free unmanaged resources
    // ~Theme()
    // {
    //     // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
    //     Dispose(disposing: false);
    // }

    public void Dispose() {
        // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
        Dispose(disposing: true);
        GC.SuppressFinalize(this);
    }

    public override bool Equals(object? obj) => Equals(obj as Theme);
    public bool Equals(Theme? other) => other is not null && Name == other.Name;
    public override int GetHashCode() => HashCode.Combine(Name);

    public static bool operator ==(Theme? left, Theme? right) => EqualityComparer<Theme>.Default.Equals(left, right);
    public static bool operator !=(Theme? left, Theme? right) => !(left == right);
}
