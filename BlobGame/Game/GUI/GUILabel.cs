using BlobGame.ResourceHandling;
using Raylib_CsLo;
using System.Numerics;

namespace BlobGame.Game.GUI;
internal sealed class GUILabel {
    private string Text { get; }
    private int FontSize { get; }

    private Vector2 TextPosition { get; }

    public GUILabel(Vector2 pos, Vector2 size, string text, Vector2? pivot = null)
        : this(pos.X, pos.Y, size.X, size.Y, text, pivot) {
    }

    public GUILabel(float x, float y, float w, float h, string text, Vector2? pivot = null) {
        if (pivot != null) {
            x += -w * pivot.Value.X;
            y += -h * pivot.Value.Y;
        }

        Text = text;
        FontSize = (int)(h * 0.7f);
        Vector2 textSize = Raylib.MeasureTextEx(ResourceManager.DefaultFont.Resource, text, FontSize, FontSize / 16f);
        TextPosition = new Vector2(x + w / 2 - textSize.X / 2f, y + h / 2 - FontSize / 2);
    }

    internal void Draw() {
        Raylib.DrawTextEx(ResourceManager.DefaultFont.Resource, Text, TextPosition, FontSize, FontSize / 16f, Raylib.WHITE);
    }

}
