using BlobGame.ResourceHandling.Resources;
using BlobGame.Util;
using Raylib_CsLo;
using System.Diagnostics;
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
    private Dictionary<string, Color> Colors { get; }

    /// <summary>
    /// The zip archive containing all assets.
    /// </summary>
    private ZipArchive? ThemeArchive { get; set; }

    /// <summary>
    /// A collection of all music data from this theme.
    /// This has to exist because Raylib.loadMusicStreamFromMemory
    /// streams in the music while playing. This just keeps it from
    /// being garbage collected.
    /// </summary>
    private List<byte[]> _MusicBuffers;

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
        Colors = new Dictionary<string, Color>();
        _MusicBuffers = new();

        WasLoaded = false;
    }

    internal void Load() {
        if (WasLoaded)
            throw new InvalidOperationException("Theme was already loaded.");

        Debug.WriteLine($"Loading theme {Name}");
        MemoryStream ms = new MemoryStream();
        using FileStream fs = new FileStream(ThemeFilePath, FileMode.Open);

        fs.CopyTo(ms);
        ms.Position = 0;

        ThemeArchive = new ZipArchive(ms, ZipArchiveMode.Read);

        ZipArchiveEntry? colorEntry = ThemeArchive.GetEntry("colors.json");
        if (colorEntry == null) {
            Debug.WriteLine($"Theme {ThemeFilePath} doesn't contain colors.");
            return;
        }

        StreamReader colorStreamReader = new StreamReader(colorEntry.Open());
        Dictionary<string, int[]>? colors = JsonSerializer.Deserialize<Dictionary<string, int[]>>(colorStreamReader.ReadToEnd());
        if (colors == null) {
            Debug.WriteLine($"colors.json in theme {ThemeFilePath} has a wrong format.");
            return;
        }

        foreach (KeyValuePair<string, int[]> entry in colors) {
            Colors[entry.Key] = new Color((byte)entry.Value[0], (byte)entry.Value[1], (byte)entry.Value[2], (byte)entry.Value[3]);
        }

        WasLoaded = true;
    }

    internal void Unload() {

    }

    internal bool DoesColorExist(string key) {
        return Colors.ContainsKey(key);
    }

    /// <summary>
    /// Tries to get a color to the given key.
    /// </summary>
    /// <param name="key"></param>
    /// <exception cref="InvalidOperationException">Thrown if the theme was not loaded.</exception>
    internal Color? GetColor(string key) {
        if (!WasLoaded)
            throw new InvalidOperationException("Theme was not loaded.");

        if (!Colors.ContainsKey(key)) {
            return null;
        }

        return Colors[key];
    }

    internal bool DoesFontExist(string key) {
        throw new NotImplementedException();
    }

    /// <summary>
    /// Tries to load a font from the zip archive.
    /// </summary>
    /// <param name="key"></param>
    /// <exception cref="InvalidOperationException">Thrown if the theme was not loaded.</exception>
    public Font? LoadFont(string key) {
        if (!WasLoaded)
            throw new InvalidOperationException("Theme was not loaded.");

        string path = $"Fonts/{key}.ttf";
        ZipArchiveEntry? zippedFont = ThemeArchive!.GetEntry(path);

        if (zippedFont == null) {
            path = $"Fonts/{key}.otf";
            zippedFont = ThemeArchive!.GetEntry(path);
        }

        if (zippedFont == null) {
            Debug.WriteLine($"Font {key} doesn't exist in this theme");
            return null;
        }

        using Stream fontStream = zippedFont.Open();
        byte[] fontData;
        using (MemoryStream ms = new MemoryStream()) {
            fontStream.CopyTo(ms);
            fontData = ms.ToArray();
        }

        Font font;
        unsafe {
            fixed (byte* fontPtr = fontData) {
                font = Raylib.LoadFontFromMemory(".ttf", fontPtr, fontData.Length, 200, null, 0);
            }
        }

        if (font.texture.id == 0) {
            Debug.WriteLine($"Failed to load font {key} from {path}");
            return null;
        }
        return font;
    }

    internal bool DoesTextureExist(string key) {
        string path = $"Textures/{key}.png";
        return ThemeArchive!.GetEntry(path) != null;
    }

    /// <summary>
    /// Tries to load a texture from the zip archive.
    /// </summary>
    /// <param name="key"></param>
    /// <exception cref="InvalidOperationException">Thrown if the theme was not loaded.</exception>
    public Texture? LoadTexture(string key) {
        if (!WasLoaded)
            throw new InvalidOperationException("Theme was not loaded.");

        string path = $"Textures/{key}.png";
        ZipArchiveEntry? zippedTexture = ThemeArchive!.GetEntry(path);

        if (zippedTexture == null) {
            Debug.WriteLine($"Texture {key} doesn't exist in this theme");
            return null;
        }

        using Stream textureStream = zippedTexture.Open();
        byte[] textureData;
        using (MemoryStream ms = new MemoryStream()) {
            textureStream.CopyTo(ms);
            ms.Position = 0;
            textureData = ms.ToArray();
        }

        Texture texture;
        unsafe {
            fixed (byte* texturePtr = textureData) {
                texture = Raylib.LoadTextureFromImage(Raylib.LoadImageFromMemory(".png", texturePtr, textureData.Length));
            }
        }

        if (texture.id == 0) {
            Debug.WriteLine($"Failed to load texture {key} from {path}");
            return null;
        }
        return texture;
    }

    internal bool DoesSoundExist(string key) {
        string path = $"Sounds/{key}.wav";
        return ThemeArchive!.GetEntry(path) != null;
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
            Debug.WriteLine($"Sound {key} doesn't exist in this theme");
            return null;
        }

        using Stream soundStream = zippedSound.Open();
        byte[] soundData;
        using (MemoryStream ms = new MemoryStream()) {
            soundStream.CopyTo(ms);
            ms.Position = 0;
            soundData = ms.ToArray();
        }

        Sound sound;
        unsafe {
            fixed (byte* soundPtr = soundData) {
                sound = Raylib.LoadSoundFromWave(Raylib.LoadWaveFromMemory(".wav", soundPtr, soundData.Length));
            }
        }

        return sound;
    }

    internal bool DoesMusicExist(string key) {
        string path = $"Music/{key}.mp3";
        return ThemeArchive!.GetEntry(path) != null;
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
            Debug.WriteLine($"Music {key} doesn't exist in this theme");
            return null;
        }

        using Stream musicStream = zippedSound.Open();
        byte[] musicData;
        using (MemoryStream ms = new MemoryStream()) {
            musicStream.CopyTo(ms);
            ms.Position = 0;
            musicData = ms.ToArray();
        }

        Music music;
        unsafe {
            fixed (byte* soundPtr = musicData) {
                music = Raylib.LoadMusicStreamFromMemory(".mp3", soundPtr, musicData.Length);
            }
        }

        // force the data to stay alive until the theme changes.
        _MusicBuffers.Add(musicData);
        music.looping = false;

        return music;
    }

    internal bool DoesTextExist(string key) {
        string path = $"Texts/{key}.json";
        return ThemeArchive!.GetEntry(path) != null;
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
            Debug.WriteLine($"Text {key} doesn't exist in this theme");
            return null;
        }

        using Stream textStream = zippedText.Open();
        Dictionary<string, string>? dict = JsonSerializer.Deserialize<Dictionary<string, string>>(textStream);

        return dict;
    }

    internal bool DoesNPatchTextureExist(string key) {
        string path = $"Textures/NPatchData/{key}.json";
        return DoesTextureExist(key) && ThemeArchive!.GetEntry(path) != null;
    }

    /// <summary>
    /// Tries to load a NPatchTexture from the zip archive.
    /// </summary>
    /// <param name="key"></param>
    /// <exception cref="InvalidOperationException">Thrown if the theme was not loaded.</exception>
    public NPatchTexture? LoadNPatchTexture(string key) {
        if (!WasLoaded)
            throw new InvalidOperationException("Theme was not loaded.");

        Texture texture = (Texture)LoadTexture(key);

        string path = $"Textures/NPatchData/{key}.json";
        ZipArchiveEntry? zippedText = ThemeArchive!.GetEntry(path);

        if (zippedText == null) {
            Debug.WriteLine($"NPatchData {key} doesn't exist in this theme");
            return null;
        }

        using Stream textStream = zippedText.Open();
        Dictionary<string, int>? dict = JsonSerializer.Deserialize<Dictionary<string, int>>(textStream);

        if (dict == null)
            return null;

        return new NPatchTexture(texture, dict["left"], dict["right"], dict["top"], dict["bottom"]);
    }

    internal bool DoesTextureAtlasExist(string key) {
        string path = $"Textures/TextureAtlasData/{key}.json";
        return DoesTextureExist(key) && ThemeArchive!.GetEntry(path) != null;
    }

    /// <summary>
    /// Tries to load a TextureAtlas from the zip archive.
    /// </summary>
    /// <param name="key"></param>
    /// <exception cref="InvalidOperationException">Thrown if the theme was not loaded.</exception>
    public TextureAtlas? LoadTextureAtlas(string key) {
        if (!WasLoaded)
            throw new InvalidOperationException("Theme was not loaded.");

        Texture texture = (Texture)LoadTexture(key);

        string path = $"Textures/TextureAtlasData/{key}.json";
        ZipArchiveEntry? zippedText = ThemeArchive!.GetEntry(path);

        if (zippedText == null) {
            Debug.WriteLine($"TextureAtlasData {key} doesn't exist in this theme");
            return null;
        }

        using Stream textStream = zippedText.Open();
        Dictionary<string, string>? dict = JsonSerializer.Deserialize<Dictionary<string, string>>(textStream);

        if (dict == null)
            return null;

        Dictionary<string, (int x, int y, int w, int h)> subTextures = new();
        foreach (KeyValuePair<string, string> item in dict) {
            string id = item.Key;
            string[] components = item.Value.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

            if (components.Length != 4) {
                Log.WriteLine($"TextureAtlasData {key} has an invalid format.", eLogType.Error);
                continue;
            }

            if (!int.TryParse(components[0], CultureInfo.InvariantCulture, out int x) ||
                !int.TryParse(components[1], CultureInfo.InvariantCulture, out int y) ||
                !int.TryParse(components[2], CultureInfo.InvariantCulture, out int w) ||
                !int.TryParse(components[3], CultureInfo.InvariantCulture, out int h)) {
                Log.WriteLine($"TextureAtlasData {key} has an invalid format.", eLogType.Error);
                continue;
            }

            subTextures[id] = (x, y, w, h);
        }

        return new TextureAtlas(texture, subTextures);
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
