using BlobGame.ResourceHandling;
using BlobGame.ResourceHandling.Resources;
using BlobGame.Util;
using Raylib_CsLo;
using System.Numerics;

namespace BlobGame.Game.Gui;
internal sealed class GuiPanel : GuiElement {
    public ColorResource Color { get; set; }
    public ColorResource AccentColor { get; set; }

    public GuiPanel(string boundsString, Vector2? pivot = null)
        : this(GuiBoundsParser.Parse(boundsString), pivot) {
    }

    public GuiPanel(Rectangle bounds, Vector2? pivot = null)
        : this(bounds.X, bounds.Y, bounds.width, bounds.height, pivot) {
    }

    public GuiPanel(float x, float y, float w, float h, Vector2? pivot = null)
        : base(x, y, w, h, pivot) {

        Color = ResourceManager.ColorLoader.Get("light_accent");
        AccentColor = ColorResource.WHITE;
    }

    protected override void DrawInternal() {
        Raylib.DrawRectangleRounded(Bounds, 0.15f, 10, Color.Resource);
        Raylib.DrawRectangleRoundedLines(Bounds, 0.15f, 10, 8, AccentColor.Resource);
    }

}
