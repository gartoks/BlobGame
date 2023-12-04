using BlobGame.ResourceHandling;
using BlobGame.Util;
using OpenTK.Mathematics;
using SimpleGL.Graphics;
using SimpleGL.Graphics.Rendering;
using SimpleGL.Graphics.Textures;

namespace BlobGame.Game.Gui;
internal sealed class GuiNPatchPanel : GuiElement {

    private string _TextureKey { get; set; }
    public string TextureKey {
        get => _TextureKey;
        set {
            if (ResourceManager.NPatchLoader.GetResourceState(value) != eResourceLoadStatus.Loaded)
                throw new ArgumentException($"Texture with key \"{value}\" is not loaded.");

            _TextureKey = value;
            Texture = null;
        }
    }
    private NPatchTexture? Texture { get; set; }
    private NPatchSprite PanelSprite { get; set; }

    public Color4 Tint { get; set; }

    public GuiNPatchPanel(string boundsString, string textureKey, int zIndex, Vector2? pivot = null)
        : this(GuiBoundsParser.Parse(boundsString), textureKey, zIndex, pivot) {
    }

    public GuiNPatchPanel(Box2 bounds, string textureKey, int zIndex, Vector2? pivot = null)
        : this(bounds.Min.X, bounds.Min.Y, bounds.Size.X, bounds.Size.Y, textureKey, zIndex, pivot) {
    }

    public GuiNPatchPanel(float x, float y, float w, float h, string textureKey, int zIndex, Vector2? pivot = null)
        : base(x, y, w, h, pivot, zIndex) {

        TextureKey = textureKey;
        Tint = Color4.White;
    }

    internal override void Load() {
        base.Load();

        Texture = ResourceManager.NPatchLoader.GetResource(TextureKey);
        PanelSprite = new NPatchSprite(Texture, GraphicsHelper.CreateDefaultTexturedShader());
    }

    protected override void DrawInternal() {
        if (Texture == null) {
            Texture = ResourceManager.NPatchLoader.GetResource(TextureKey);
            PanelSprite.Texture = Texture;
        }

        PanelSprite.Transform.Pivot = Pivot;
        PanelSprite.Transform.Position = Bounds.Min;
        PanelSprite.Transform.Scale = Bounds.Size;
        PanelSprite.Transform.Rotation = 0;
        PanelSprite.Transform.ZIndex = ZIndex;
        PanelSprite.Render();
    }

}
