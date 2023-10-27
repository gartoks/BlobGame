using Raylib_CsLo;
using System.Diagnostics;
using System.IO.Compression;
using System.Text.Json;

namespace BlobGame.ResourceHandling;
/// <summary>
/// Class for one set of game resources. Doesn't cache anything.
/// </summary>
internal class Theme {
    /// <summary>
    /// The zip archive containing all assets.
    /// </summary>
    private ZipArchive Assets;
    /// <summary>
    /// A mapping of color name to colors.
    /// </summary>
    private Dictionary<string, Color> Colors;

    /// <summary>
    /// Constructor to load a theme from disk.
    /// </summary>
    public Theme(string path) {
        Assets = new ZipArchive(new FileStream(path, FileMode.Open));

        Colors = new Dictionary<string, Color>();

        ZipArchiveEntry? colorEntry = Assets.GetEntry("colors.json");
        if (colorEntry == null) {
            Debug.WriteLine($"Theme {path} doesn't contain colors.");
            return;
        }

        StreamReader colorStreamReader = new StreamReader(colorEntry.Open());
        Dictionary<string, int[]>? colorsInWrongFormat = JsonSerializer.Deserialize<Dictionary<string, int[]>>(colorStreamReader.ReadToEnd());
        if (colorsInWrongFormat == null) {
            Debug.WriteLine($"colors.json in theme {path} has a wrong format.");
            return;
        }

        Colors = colorsInWrongFormat
            .ToDictionary(pair => pair.Key, pair => new Color((byte)pair.Value[0], (byte)pair.Value[1], (byte)pair.Value[2], (byte)pair.Value[3]));
    }


    /// <summary>
    /// Tries to load a font from the zip archive.
    /// </summary>
    /// <param name="key"></param>
    public Font? LoadFont(string key) {
        string path = $"Fonts/{key}.ttf";
        ZipArchiveEntry? zippedFont = Assets.GetEntry(path);

        if (zippedFont == null) {
            Debug.WriteLine($"Font {key} doesn't exist in this theme");
            return null;
        }

        Stream fontStream = zippedFont.Open();
        byte[] fontData;
        using (MemoryStream ms = new MemoryStream()) {
            fontStream.CopyTo(ms);
            fontData = ms.ToArray();
        }

        Font font;
        unsafe {
            fixed (byte* fontPtr = fontData) {
                font = Raylib.LoadFontFromMemory("ttf", fontPtr, fontData.Length, 200, null, 0);
            }
        }

        if (font.texture.id == 0) {
            Debug.WriteLine($"Failed to load font {key} from {path}");
            return null;
        }
        return font;
    }

    /// <summary>
    /// Tries to load a texture from the zip archive.
    /// </summary>
    /// <param name="key"></param>
    public Texture? LoadTexture(string key) {
        string path = $"Textures/{key}.png";
        ZipArchiveEntry? zippedTexture = Assets.GetEntry(path);

        if (zippedTexture == null) {
            Debug.WriteLine($"Texture {key} doesn't exist in this theme");
            return null;
        }

        Stream textureStream = zippedTexture.Open();
        byte[] textureData;
        using (MemoryStream ms = new MemoryStream()) {
            textureStream.CopyTo(ms);
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

    /// <summary>
    /// Tries to load a sound from the zip archive.
    /// </summary>
    /// <param name="key"></param>
    public Sound? LoadSound(string key) {
        string path = $"Sounds/{key}.wav";
        ZipArchiveEntry? zippedSound = Assets.GetEntry(path);

        if (zippedSound == null) {
            Debug.WriteLine($"Sound {key} doesn't exist in this theme");
            return null;
        }

        Stream soundStream = zippedSound.Open();
        byte[] soundData;
        using (MemoryStream ms = new MemoryStream()) {
            soundStream.CopyTo(ms);
            soundData = ms.ToArray();
        }

        Sound sound;
        unsafe {
            fixed (byte* soundPtr = soundData) {
                sound = Raylib.LoadSoundFromWave(Raylib.LoadWaveFromMemory("wav", soundPtr, soundData.Length));
            }
        }

        return sound;
    }

    /// <summary>
    /// Tries to load a music from the zip archive.
    /// </summary>
    /// <param name="key"></param>
    public Music? LoadMusic(string key) {
        string path = $"Music/{key}.wav";
        ZipArchiveEntry? zippedSound = Assets.GetEntry(path);

        if (zippedSound == null) {
            Debug.WriteLine($"Music {key} doesn't exist in this theme");
            return null;
        }

        Stream musicStream = zippedSound.Open();
        byte[] musicData;
        using (MemoryStream ms = new MemoryStream()) {
            musicStream.CopyTo(ms);
            musicData = ms.ToArray();
        }

        Music music;
        unsafe {
            fixed (byte* soundPtr = musicData) {
                music = Raylib.LoadMusicStreamFromMemory("wav", soundPtr, musicData.Length);
            }
        }

        return music;
    }

    /// <summary>
    /// Tries to get a color to the given key.
    /// </summary>
    /// <param name="key"></param>
    public Color? GetColor(string key) {
        if (!Colors.ContainsKey(key)) {
            return null;
        }

        return Colors[key];
    }
}
