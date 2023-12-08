using BlobGame.App;
using BlobGame.Util;
using Raylib_CsLo;
using System.Numerics;

namespace BlobGame.Game.Gui;
internal sealed class GuiTextButton : InteractiveGuiElement {
    public GuiPanel Panel { get; }
    public GuiLabel Label { get; }

    public bool IsClicked { get; private set; }

    private string BaseTexture { get; set; }
    private string SelectedTexture { get; set; }

    public GuiTextButton(string boundsString, string text, Vector2? pivot = null)
        : this(GuiBoundsParser.Parse(boundsString), text, pivot) {
    }

    private GuiTextButton(Rectangle bounds, string text, Vector2? pivot = null)
        : this(bounds.X, bounds.Y, bounds.width, bounds.height, text, pivot) {
    }

    public GuiTextButton(float x, float y, float w, float h, string text, Vector2? pivot = null)
        : base(x, y, w, h, pivot) {

        BaseTexture = "button_up";
        SelectedTexture = "button_selected";
        Panel = new GuiPanel(Bounds, BaseTexture, new Vector2(0, 0));
        Label = new GuiLabel(Bounds, text, new Vector2(0, 0));
    }

    internal override void Load() {
    }

    protected override void DrawInternal() {
        string texture = BaseTexture;
        if (IsHovered)
            texture = SelectedTexture;

        IsClicked = (IsHovered && Input.IsMouseButtonActive(MouseButton.MOUSE_BUTTON_LEFT)) ||
            (GuiManager.HasFocus(this) && Input.IsHotkeyActive("confirm"));

        if (IsClicked)
            Focus();

        if (HasFocus())
            texture = SelectedTexture;

        Panel.TextureKey = texture;
        Panel.Draw();

        Label.Draw();

        Input.WasMouseHandled[MouseButton.MOUSE_BUTTON_LEFT] |= IsClicked;
    }

}
