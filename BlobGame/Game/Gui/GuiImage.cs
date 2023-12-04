using BlobGame.ResourceHandling;
using OpenTK.Mathematics;
using SimpleGL.Graphics.Rendering;
using SimpleGL.Graphics.Textures;

namespace BlobGame.Game.Gui;
internal sealed class GUIImage : GuiElement {
    private string _TextureKey { get; set; }
    public string TextureKey {
        get => _TextureKey;
        set {
            _TextureKey = value;
            Texture = null;
        }
    }
    private Texture? Texture { get; set; }

    public Color4 Tint { get; set; }

    public float Scale { get; set; }

    public GUIImage(float x, float y, float width, float height, int zIndex, string textureKey, Vector2? pivot = null)
        : base(x, y, width, height, pivot, zIndex) {

        TextureKey = textureKey;
        Tint = Color4.White;

        Scale = 1;
        Texture = null;
    }

    protected override void DrawInternal() {
        if (Texture == null)
            Texture = ResourceManager.TextureLoader.GetResource(TextureKey);

        Primitives.DrawSprite(Bounds.Min, Bounds.Size * Scale, Pivot, 0, ZIndex, Texture, Tint);
    }

}
