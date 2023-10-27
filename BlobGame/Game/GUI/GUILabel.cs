using BlobGame.Drawing;
using Raylib_CsLo;
using System.Numerics;

namespace BlobGame.Game.Gui;
internal sealed class GuiLabel {
    private string Text { get; }
    private int FontSize { get; }

    private Vector2 TextPosition { get; }

    public GuiLabel(Vector2 pos, Vector2 size, string text, Vector2? pivot = null)
        : this(pos.X, pos.Y, size.X, size.Y, text, pivot) {
    }

    public GuiLabel(float x, float y, float w, float h, string text, Vector2? pivot = null) {
        if (pivot != null) {
            x += -w * pivot.Value.X;
            y += -h * pivot.Value.Y;
        }

        Text = text;
        FontSize = (int)(h * 0.6f);
        Vector2 textSize = Raylib.MeasureTextEx(Renderer.Font.Resource, text, FontSize, FontSize / 16f);
        TextPosition = new Vector2(x + w / 2 - textSize.X / 1.75f, y + h / 2 - FontSize / 2);
    }

    internal void Draw() {
        Raylib.DrawTextEx(Renderer.Font.Resource, Text, TextPosition, FontSize, FontSize / 16f, Raylib.WHITE);
    }

}
