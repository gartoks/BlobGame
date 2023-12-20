using BlobGame.ResourceHandling;
using BlobGame.ResourceHandling.Resources;
using BlobGame.Util;
using Raylib_CsLo;
using System.Numerics;

namespace BlobGame.Game.Gui;
internal sealed class GuiPanel : GuiElement {

    private string _TextureKey { get; set; }
    public string TextureKey {
        get => _TextureKey;
        set {
            _TextureKey = value;
            Texture = null;
        }
    }
    private NPatchTextureResource? Texture { get; set; }

    public ColorResource Tint { get; set; }

    public GuiPanel(string boundsString, string textureKey, Vector2? pivot = null)
        : this(GuiBoundsParser.Parse(boundsString), textureKey, pivot) {
    }

    public GuiPanel(Rectangle bounds, string textureKey, Vector2? pivot = null)
        : this(bounds.X, bounds.Y, bounds.width, bounds.height, textureKey, pivot) {
    }

    public GuiPanel(float x, float y, float w, float h, string textureKey, Vector2? pivot = null)
        : base(x, y, w, h, pivot) {

        TextureKey = textureKey;
        Tint = ColorResource.WHITE;
    }

    internal override void Load() {
        base.Load();

        Texture = ResourceManager.NPatchTextureLoader.Get(TextureKey);
    }

    protected override void DrawInternal() {
        if (Texture == null)
            Texture = ResourceManager.NPatchTextureLoader.Get(TextureKey);

        Texture.Draw(Bounds, Pivot, 0, Tint.Resource);
    }

}
