using BlobGame.App;
using BlobGame.ResourceHandling.Resources;
using BlobGame.Util;
using Raylib_CsLo;
using System.Numerics;

namespace BlobGame.Game.Gui;
internal class GuiTextbox : InteractiveGuiElement {
    private const float TEXT_SPACING = 20;
    private const string ALLOWED_CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!\"§$%&/()=?*+~#'-_.:,;<>|^°@€ ";

    public string Text {
        get => Label.Text;
        set {
            Label.Text = value;

            if (Label.GetTextSize().X > Bounds.width - 2 * TEXT_SPACING)
                Label.Text = Label.Text[..^1];
        }
    }

    public ColorResource TextColor {
        get => Label.Color;
        set => Label.Color = value;
    }

    private GuiNPatchPanel Panel { get; }
    private GuiLabel Label { get; }

    public Predicate<char>? CharFilter { get; init; }

    private string BaseTexture { get; set; }
    private string SelectedTexture { get; set; }

    public GuiTextbox(string boundsString, Vector2? pivot = null)
        : this(GuiBoundsParser.Parse(boundsString), pivot) {
    }

    private GuiTextbox(Rectangle bounds, Vector2? pivot = null)
        : this(bounds.X, bounds.Y, bounds.width, bounds.height, pivot) {
    }

    public GuiTextbox(float x, float y, float w, float h, Vector2? pivot = null)
        : base(x, y, w, h, pivot) {

        BaseTexture = "button_up";
        SelectedTexture = "button_selected";

        Panel = new GuiNPatchPanel(x, y, w, h, BaseTexture, new Vector2(0, 0));
        Label = new GuiLabel(x + TEXT_SPACING, y, w - 2 * TEXT_SPACING, h, string.Empty, new Vector2(0, 0));
        Label.TextAlignment = eTextAlignment.Left;
    }

    protected override void DrawInternal() {
        bool shouldFocus = IsHovered && Input.IsMouseButtonActive(MouseButton.MOUSE_BUTTON_LEFT);

        if (shouldFocus)
            Focus();

        if (HasFocus()) {
            if (Raylib.IsKeyReleased(KeyboardKey.KEY_BACKSPACE) && Label.Text.Length > 0)
                Text = Label.Text[..^1];
            else {
                char c = (char)Raylib.GetCharPressed();
                if (ALLOWED_CHARS.Contains(c) && (CharFilter == null || CharFilter(c)))
                    Text += c.ToString();
            }
        }

        string texture = BaseTexture;
        if (HasFocus())
            texture = SelectedTexture;
        Panel.TextureKey = texture;

        Panel.Draw();
        Label.Draw();
    }

}
