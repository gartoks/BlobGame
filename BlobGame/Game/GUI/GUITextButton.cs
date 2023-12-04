using BlobGame.App;
using BlobGame.Util;
using OpenTK.Mathematics;
using OpenTK.Windowing.GraphicsLibraryFramework;

namespace BlobGame.Game.Gui;
internal sealed class GuiTextButton : InteractiveGuiElement {
    public string BaseTexture { get; set; }
    public string SelectedTexture { get; set; }

    public GuiNPatchPanel Panel { get; }
    public GuiLabel Label { get; }

    public bool IsClicked { get; private set; }

    public GuiTextButton(string boundsString, string text, int zIndex, Vector2? pivot = null)
        : this(GuiBoundsParser.Parse(boundsString), text, zIndex, pivot) {
    }

    private GuiTextButton(Box2 bounds, string text, int zIndex, Vector2? pivot = null)
        : this(bounds.X(), bounds.Y(), bounds.Width(), bounds.Height(), text, zIndex, pivot) {
    }

    public GuiTextButton(float x, float y, float w, float h, string text, int zIndex, Vector2? pivot = null)
        : base(x, y, w, h, zIndex, pivot) {

        BaseTexture = "button_up";
        SelectedTexture = "button_selected";
        Panel = new GuiNPatchPanel(Bounds, BaseTexture, zIndex, new Vector2(0, 0));
        Label = new GuiLabel(Bounds, text, zIndex + 1, new Vector2(0, 0));
    }

    internal override void Load() {
        base.Load();

        Panel.Load();
        Label.Load();
    }

    protected override void DrawInternal() {
        string texture = BaseTexture;
        if (IsHovered)
            texture = SelectedTexture;

        IsClicked = (IsHovered && Input.IsMouseButtonActive(MouseButton.Left)) ||
            (GuiManager.HasFocus(this) && Input.IsHotkeyActive("confirm"));

        if (IsClicked)
            Focus();

        if (HasFocus())
            texture = SelectedTexture;

        Panel.TextureKey = texture;
        Panel.Draw();

        Label.Draw();

        Input.WasMouseHandled[MouseButton.Left] |= IsClicked;
    }

}
