using Raylib_CsLo;
using System.Numerics;

namespace BlobGame.Game.Gui;
internal sealed class GuiPanel {
    public Color Color { get; set; }

    private Rectangle Bounds { get; }

    public GuiPanel(Vector2 pos, Vector2 size, Color color, Vector2? pivot = null)
        : this(pos.X, pos.Y, size.X, size.Y, color, pivot) {
    }

    public GuiPanel(float x, float y, float w, float h, Color color, Vector2? pivot = null) {
        if (pivot != null) {
            x += -w * pivot.Value.X;
            y += -h * pivot.Value.Y;
        }

        Color = color;
        Bounds = new Rectangle(x, y, w, h);
    }

    internal void Draw() {
        Raylib.DrawRectangleRounded(Bounds, 0.15f, 10, Color);
        Raylib.DrawRectangleRoundedLines(Bounds, 0.15f, 10, 8, Raylib.WHITE);
    }

}
