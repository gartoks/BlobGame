using BlobGame.ResourceHandling;
using BlobGame.Util;
using OpenTK.Mathematics;
using SimpleGL.Graphics.Rendering;

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

    //public bool DrawOutline { get; set; }
    //public Color4 OutlineColor { get; set; }

    private float FontSize { get; }
    private int FontSizeInt => (int)FontSize;
    //private float FontSpacing { get; }

    private Vector2 TextPosition { get; set; }

    private MeshFont? LastUsedFont { get; set; }

    public GuiLabel(string boundsString, string text, int zIndex, Vector2? pivot = null)
        : this(GuiBoundsParser.Parse(boundsString), text, zIndex, pivot) {
    }

    public GuiLabel(Box2 bounds, string text, int zIndex, Vector2? pivot = null)
        : this(bounds.Min.X, bounds.Min.Y, bounds.Width(), bounds.Height(), text, zIndex, pivot) {
    }

    public GuiLabel(float x, float y, float w, float h, string text, int zIndex, Vector2? pivot = null)
        : base(x, y, w, h, pivot, zIndex) {

        _Text = text;
        FontSize = h * 0.6f; // TODO check if fits
        //FontSpacing = FontSize / 64f;
        TextAlignment = eTextAlignment.Center;

        //DrawOutline = false;
        //OutlineColor = ResourceManager.ColorLoader.GetResource("outline");
    }

    protected override void DrawInternal() {
        MeshFont font = Fonts.GetGuiFont(FontSizeInt);
        if (LastUsedFont == null || !LastUsedFont.Equals(font))  // TODO Check if fits
            CalculateTextPosition();
        LastUsedFont = font;


        /*if (DrawOutline) {
            Vector2 textSize = GetTextSize();
            Vector2 outlineSize = Raylib.MeasureTextEx(RenderManager.GuiFont.Resource, Text, FontSize * 1.05f, FontSpacing);
            Vector2 outlineOffset = new Vector2(
                (outlineSize.X - textSize.X) / 2f,
                (outlineSize.Y - textSize.Y) / 2f
                );

            Raylib.DrawTextEx(RenderManager.GuiFont.Resource, Text, TextPosition - outlineOffset, FontSize * 1.05f, FontSpacing, OutlineColor.Resource);
        }*/

        Primitives.DrawText(font, Text, Color4.White, TextPosition, new Vector2(0.5f, 0.5f), 0, ZIndex);
    }

    internal Vector2 GetTextSize() {
        MeshFont font = Fonts.GetGuiFont(FontSizeInt);
        return font.MeasureText(Text);
    }

    private void CalculateTextPosition() {
        Vector2 textSize = GetTextSize();
        switch (_TextAlignment) {
            case eTextAlignment.Center:
                TextPosition = new Vector2(Bounds.X() + (Bounds.Width() - textSize.X) / 2f, Bounds.Y() + (Bounds.Height() - FontSize) / 2f);
                break;
            case eTextAlignment.Left:
                TextPosition = new Vector2(Bounds.X(), Bounds.Y() + (Bounds.Height() - FontSize) / 2f);
                break;
            case eTextAlignment.Right:
                TextPosition = new Vector2(Bounds.X() + Bounds.Width() - textSize.X, Bounds.Y() + (Bounds.Height() - FontSize) / 2f);
                break;
        }
    }
}
