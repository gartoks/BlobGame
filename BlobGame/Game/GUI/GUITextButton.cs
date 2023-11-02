using BlobGame.App;
using BlobGame.ResourceHandling;
using BlobGame.Util;
using Raylib_CsLo;
using System.Numerics;

namespace BlobGame.Game.Gui;
internal sealed class GuiTextButton : InteractiveGuiElement {
    public GuiPanel Panel { get; }
    public GuiLabel Label { get; }

    public bool IsClicked { get; private set; }

    public GuiTextButton(string boundsString, string text, Vector2? pivot = null)
        : this(GuiBoundsParser.Parse(boundsString), text, pivot) {
    }

    private GuiTextButton(Rectangle bounds, string text, Vector2? pivot = null)
        : this(bounds.X, bounds.Y, bounds.width, bounds.height, text, pivot) {
    }

    public GuiTextButton(float x, float y, float w, float h, string text, Vector2? pivot = null)
        : base(x, y, w, h, pivot) {

        Panel = new GuiPanel(Bounds, new Vector2(0, 0));
        Label = new GuiLabel(Bounds, text, new Vector2(0, 0));
    }

    protected override void DrawInternal() {
        ColorResource bgColor = ResourceManager.GetColor("light_accent");
        if (IsHovered)
            bgColor = ResourceManager.GetColor("dark_accent");

        Panel.Color = bgColor;

        IsClicked = (IsHovered && Input.IsMouseButtonActive(MouseButton.MOUSE_BUTTON_LEFT)) ||
            (GuiManager.HasFocus(this) && Input.IsHotkeyActive("confirm"));

        if (IsClicked)
            Focus();

        ColorResource accentColor = ColorResource.WHITE;
        if (HasFocus())
            accentColor = ResourceManager.GetColor("highlight");
        Panel.AccentColor = accentColor;

        Panel.Draw();
        Label.Draw();

        Input.WasMouseHandled[MouseButton.MOUSE_BUTTON_LEFT] |= IsClicked;
    }

}
