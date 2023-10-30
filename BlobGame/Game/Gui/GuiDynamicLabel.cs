using BlobGame.Drawing;
using Raylib_CsLo;
using System.Numerics;

namespace BlobGame.Game.Gui;
internal sealed class GuiDynamicLabel {
    private string Text { get; }
    private float FontSize { get; }

    private Vector2 TextPosition { get; }

    public GuiDynamicLabel(Vector2 pos, string text, float fontSize, Vector2? pivot = null)
        : this(pos.X, pos.Y, text, fontSize, pivot) {
    }

    public GuiDynamicLabel(float x, float y, string text, float fontSize, Vector2? pivot = null) {
        Vector2 textSize = Raylib.MeasureTextEx(Renderer.Font.Resource, text, FontSize, FontSize / 16f);

        if (pivot != null) {
            x += -textSize.X * pivot.Value.X;
            y += -textSize.Y * pivot.Value.Y;
        }

        Text = text;
        FontSize = fontSize;
        TextPosition = new Vector2(x, y);
    }

    internal void Draw() {
        Raylib.DrawTextEx(Renderer.Font.Resource, Text, TextPosition, FontSize, FontSize / 16f, Raylib.WHITE);
    }

}
