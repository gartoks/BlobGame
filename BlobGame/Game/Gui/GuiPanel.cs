using BlobGame.ResourceHandling;
using BlobGame.Util;
using OpenTK.Mathematics;
using SimpleGL.Graphics.Rendering;

namespace BlobGame.Game.Gui;
internal sealed class GuiPanel : GuiElement {
    public Color4 Color { get; set; }
    public Color4 AccentColor { get; set; }

    public GuiPanel(string boundsString, int zIndex, Vector2? pivot = null)
        : this(GuiBoundsParser.Parse(boundsString), zIndex, pivot) {
    }

    public GuiPanel(Box2 bounds, int zIndex, Vector2? pivot = null)
        : this(bounds.X(), bounds.Y(), bounds.Width(), bounds.Height(), zIndex, pivot) {
    }

    public GuiPanel(float x, float y, float w, float h, int zIndex, Vector2? pivot = null)
        : base(x, y, w, h, pivot, zIndex) {

        Color = ResourceManager.ColorLoader.GetResource("light_accent");
        AccentColor = Color4.White;
    }

    protected override void DrawInternal() {
        Primitives.DrawRectangle(Bounds.Min, Bounds.Size, Pivot, 0, ZIndex, Color);
        Primitives.DrawRectangleLines(Bounds.Min, Bounds.Size, 8, Pivot, 0, ZIndex, AccentColor);
    }
}
