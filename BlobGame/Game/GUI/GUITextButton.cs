using BlobGame.App;
using BlobGame.Drawing;
using BlobGame.ResourceHandling;
using BlobGame.Util;
using Raylib_CsLo;
using System.Numerics;

namespace BlobGame.Game.GUI;
internal sealed class GUITextButton {
    private string Text { get; }
    private int FontSize { get; }

    private Rectangle Bounds { get; }
    private Vector2 TextPosition { get; }

    public GUITextButton(Vector2 pos, Vector2 size, string text, Vector2? pivot = null)
        : this(pos.X, pos.Y, size.X, size.Y, text, pivot) {
    }

    public GUITextButton(float x, float y, float w, float h, string text, Vector2? pivot = null) {
        if (pivot != null) {
            x += -w * pivot.Value.X;
            y += -h * pivot.Value.Y;
        }

        Text = text;
        FontSize = (int)(h * 0.7f);
        Bounds = new Rectangle(x, y, w, h);
        Vector2 textSize = Raylib.MeasureTextEx(ResourceManager.DefaultFont.Resource, text, FontSize, FontSize / 16f);
        TextPosition = new Vector2(x + w / 2 - textSize.X / 2f, y + h / 2 - FontSize / 2);
    }

    internal bool Draw() {
        bool containsMouse = Bounds.Contains(Input.ScreenToWorld(Raylib.GetMousePosition()));
        Color bgColor = Renderer.MELBA_LIGHT_PINK;
        if (containsMouse)
            bgColor = Renderer.MELBA_DARK_PINK;

        Raylib.DrawRectangleRounded(Bounds, 0.15f, 10, bgColor);
        Raylib.DrawRectangleRoundedLines(Bounds, 0.15f, 10, 8, Raylib.WHITE);
        Raylib.DrawTextEx(ResourceManager.DefaultFont.Resource, Text, TextPosition, FontSize, FontSize / 16f, Raylib.WHITE);

        return containsMouse && Input.IsMouseButtonActive(MouseButton.MOUSE_BUTTON_LEFT);
    }

}
