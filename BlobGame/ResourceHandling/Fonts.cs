using SimpleGL.Graphics;
using SimpleGL.Graphics.Rendering;
using SimpleGL.Util;

namespace BlobGame.ResourceHandling;
internal static class Fonts {
    private static Dictionary<(string name, int size), MeshFont> StoredFonts { get; } = new();

    internal static void Initialize() {
    }

    internal static void Load() {
    }

    internal static void Unload() {
        foreach (MeshFont font in StoredFonts.Values)
            font.Dispose();

        StoredFonts.Clear();
    }

    public static MeshFont GetGuiFont(int fontSize) {
        return GetFont("gui", fontSize);
    }

    public static MeshFont GetMainFont(int fontSize) {
        return GetFont("main", fontSize);
    }

    public static MeshFont GetFont(string name, int fontSize) {
        MeshFont? mFont;
        if (StoredFonts.TryGetValue((name, fontSize), out mFont))
            return mFont;

        if (ResourceManager.GetResourceState(name) is not eResourceLoadStatus.Loaded)
            throw new InvalidOperationException($"Cannot load font {name} because it is not loaded");

        FontFamilyData fontFamily = ResourceManager.FontLoader.GetResource(name);
        mFont = GraphicsHelper.CreateMeshFont(new FontData(fontFamily, fontSize), GraphicsHelper.CreateDefaultUntexturedShader());

        StoredFonts[(name, fontSize)] = mFont;
        return mFont;
    }
}
