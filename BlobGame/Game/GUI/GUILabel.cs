using BlobGame.Drawing;
using BlobGame.Util;
using Raylib_CsLo;
using System.Numerics;

namespace BlobGame.Game.Gui;

internal enum eTextAlignment { Left, Center, Right }

internal sealed class GuiLabel : GuiElement {
    private string _Text { get; set; }
    public string Text {
        get => _Text;
        set {
            _Text = value ?? string.Empty;
            CalculateTextPosition();
        }
    }

    private eTextAlignment _TextAlignment { get; set; }
    public eTextAlignment TextAlignment {
        get => _TextAlignment;
        set {
            _TextAlignment = value;
            CalculateTextPosition();
        }
    }

    private float FontSize { get; }
    private float FontSpacing { get; }

    private Vector2 TextPosition { get; set; }

    public GuiLabel(string boundsString, string text, Vector2? pivot = null)
        : this(GuiBoundsParser.Parse(boundsString), text, pivot) {
    }

    public GuiLabel(Rectangle bounds, string text, Vector2? pivot = null)
        : this(bounds.X, bounds.Y, bounds.width, bounds.height, text, pivot) {
    }

    public GuiLabel(float x, float y, float w, float h, string text, Vector2? pivot = null)
        : base(x, y, w, h, pivot) {

        _Text = text;
        FontSize = h * 0.6f;
        FontSpacing = FontSize / 16f;
        TextAlignment = eTextAlignment.Center;
    }

    protected override void DrawInternal() {
        Raylib.DrawTextEx(Renderer.Font.Resource, Text, TextPosition, FontSize, FontSpacing, Raylib.WHITE);
    }

    internal Vector2 GetTextSize() {
        return Raylib.MeasureTextEx(Renderer.Font.Resource, Text, FontSize, FontSpacing);
    }

    private void CalculateTextPosition() {
        Vector2 textSize = GetTextSize();
        switch (_TextAlignment) {
            case eTextAlignment.Center:
                TextPosition = new Vector2(Bounds.X + (Bounds.width - textSize.X) / 2f, Bounds.Y + (Bounds.height - FontSize) / 2f);
                break;
            case eTextAlignment.Left:
                TextPosition = new Vector2(Bounds.X, Bounds.Y + (Bounds.height - FontSize) / 2f);
                break;
            case eTextAlignment.Right:
                TextPosition = new Vector2(Bounds.x + Bounds.width - textSize.X, Bounds.Y + (Bounds.height - FontSize) / 2f);
                break;
        }
    }
}
