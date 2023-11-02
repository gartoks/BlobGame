using BlobGame.Drawing;
using Raylib_CsLo;
using System.Numerics;

namespace BlobGame.Game.Gui;
internal sealed class GuiDynamicLabel : GuiElement {
    public string Text { get; set; }

    private float FontSize { get; }
    private float FontSpacing { get; }

    private Vector2 TextPosition { get; }

    public GuiDynamicLabel(float x, float y, string text, float fontSize, Vector2? pivot = null)
        : base(x, y,
            Raylib.MeasureTextEx(Renderer.Font.Resource, text, fontSize, fontSize / 16f).X,
            Raylib.MeasureTextEx(Renderer.Font.Resource, text, fontSize, fontSize / 16f).Y, pivot) {

        Text = text;
        FontSize = fontSize;
        FontSpacing = FontSize / 16f;
        TextPosition = new Vector2(x, y);
    }

    protected override void DrawInternal() {
        Raylib.DrawTextEx(Renderer.Font.Resource, Text, TextPosition, FontSize, FontSpacing, Raylib.WHITE);
    }

}
