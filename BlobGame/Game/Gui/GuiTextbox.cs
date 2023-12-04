using BlobGame.App;
using BlobGame.ResourceHandling;
using BlobGame.Util;
using OpenTK.Mathematics;
using OpenTK.Windowing.Common;
using OpenTK.Windowing.GraphicsLibraryFramework;

namespace BlobGame.Game.Gui;
internal class GuiTextbox : InteractiveGuiElement {
    private const float TEXT_SPACING = 20;
    private const string ALLOWED_CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!\"§$%&/()=?*+~#'-_.:,;<>|^°@€ ";

    public string Text {
        get => Label.Text;
        set {
            Label.Text = value;

            if (Label.GetTextSize().X > Bounds.Width() - 2 * TEXT_SPACING)
                Label.Text = Label.Text[..^1];
        }
    }

    private GuiPanel Panel { get; }
    private GuiLabel Label { get; }

    public Predicate<char>? CharFilter { get; init; }

    public GuiTextbox(string boundsString, int zIndex, Vector2? pivot = null)
        : this(GuiBoundsParser.Parse(boundsString), zIndex, pivot) {
    }

    private GuiTextbox(Box2 bounds, int zIndex, Vector2? pivot = null)
        : this(bounds.X(), bounds.Y(), bounds.Width(), bounds.Height(), zIndex, pivot) {
    }

    public GuiTextbox(float x, float y, float w, float h, int zIndex, Vector2? pivot = null)
        : base(x, y, w, h, zIndex, pivot) {
        Panel = new GuiPanel(x, y, w, h, ZIndex, new Vector2(0, 0));
        Label = new GuiLabel(x + TEXT_SPACING, y, w - 2 * TEXT_SPACING, h, string.Empty, ZIndex + 1, new Vector2(0, 0));
        Label.TextAlignment = eTextAlignment.Left;

        GameApplication.Window.TextInput += OnTextInput;
        GameApplication.Window.KeyDown += OnKeyDown;
    }

    ~GuiTextbox() {
        GameApplication.Window.TextInput -= OnTextInput;
    }

    internal override void Load() {
        base.Load();

        Panel.Load();
        Label.Load();
    }

    protected override void DrawInternal() {
        bool shouldFocus = IsHovered && Input.IsMouseButtonActive(MouseButton.Left);

        if (shouldFocus)
            Focus();

        Color4 accentColor = Color4.White;
        if (HasFocus())
            accentColor = ResourceManager.ColorLoader.GetResource("highlight");
        Panel.AccentColor = accentColor;

        Panel.Draw();
        Label.Draw();
    }

    private void OnTextInput(TextInputEventArgs args) {
        char c = args.AsString.FirstOrDefault();
        if (HasFocus() && ALLOWED_CHARS.Contains(c) && (CharFilter == null || CharFilter(c)))
            Text += c.ToString();
    }

    private void OnKeyDown(KeyboardKeyEventArgs args) {
        if (HasFocus() && Label.Text.Length > 0 && args.Key == Keys.Backspace)
            Text = Label.Text[..^1];
    }
}
