using BlobGame.ResourceHandling;
using OpenTK.Mathematics;
using SimpleGL.Graphics.Rendering;

namespace BlobGame.Game.Gui;
internal sealed class GuiDynamicLabel : GuiElement {
    public string Text { get; set; }

    private float FontSize { get; }
    private int FontSizeInt => (int)FontSize;
    //private float FontSpacing { get; }

    private Vector2 TextPosition { get; }

    public GuiDynamicLabel(float x, float y, string text, float fontSize, int zIndex, Vector2? pivot = null)
        : base(new Vector2(x, y), CalculateSize(fontSize, text), pivot, zIndex) {
        Text = text;
        FontSize = fontSize;
        //FontSpacing = FontSize / 64f;
        TextPosition = new Vector2(x, y);
    }

    protected override void DrawInternal() {
        MeshFont font = Fonts.GetGuiFont(FontSizeInt);
        Primitives.DrawText(font, Text, Color4.White, TextPosition, new Vector2(0.5f, 0.5f), 0, ZIndex);
    }

    private static Vector2 CalculateSize(float fontSize, string text) {
        MeshFont font = Fonts.GetGuiFont((int)fontSize);
        return font.MeasureText(text);
    }
}
