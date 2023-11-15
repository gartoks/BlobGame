using BlobGame.App;
using BlobGame.Audio;
using BlobGame.Drawing;
using BlobGame.ResourceHandling;
using BlobGame.ResourceHandling.Resources;
using BlobGame.Util;
using Raylib_CsLo;
using System.Numerics;

namespace BlobGame.Game.Gui;
internal class GuiSelector : InteractiveGuiElement {
    private const float BUTTON_SPACING = 10;

    private IReadOnlyList<SelectionElement> Elements { get; }

    public GuiPanel Panel { get; }
    public GuiTextButton DecreaseButton { get; }
    public GuiTextButton IncreaseButton { get; }

    private int FontSize { get; }
    private float FontSpacing { get; }

    private int SelectedIndex { get; set; }
    public SelectionElement SelectedElement => Elements[SelectedIndex];

    public bool IsClicked { get; private set; }

    public GuiSelector(string boundsString, SelectionElement[] elements, int selectedIndex, Vector2? pivot = null)
        : this(GuiBoundsParser.Parse(boundsString), elements, selectedIndex, pivot) {
    }

    private GuiSelector(Rectangle bounds, SelectionElement[] elements, int selectedIndex, Vector2? pivot = null)
        : this(bounds.X, bounds.Y, bounds.width, bounds.height, elements, selectedIndex, pivot) {
    }

    private GuiSelector(float x, float y, float w, float h, SelectionElement[] elements, int selectedIndex, Vector2? pivot = null)
        : base(x, y, w, h, pivot) {
        float buttonSize = MathF.Min(w, h);

        Elements = elements;
        FontSize = (int)(buttonSize * 0.7f);
        FontSpacing = FontSize / 64f;
        //Bounds = new Rectangle(x + buttonSize + 10, y, w - 2 * buttonSize - 20, h);

        Panel = new GuiPanel(x + buttonSize + BUTTON_SPACING, y, w - 2 * buttonSize - 2 * BUTTON_SPACING, h, new Vector2(0, 0));
        DecreaseButton = new GuiTextButton(Bounds.x, y, buttonSize, buttonSize, "<", new Vector2(0, 0));
        IncreaseButton = new GuiTextButton(Bounds.x + Bounds.width, Bounds.y, buttonSize, buttonSize, ">", new Vector2(1, 0));

        SelectedIndex = selectedIndex;
    }


    protected override void DrawInternal() {
        bool shouldFocus = IsHovered && Input.IsMouseButtonActive(MouseButton.MOUSE_BUTTON_LEFT);
        if (shouldFocus)
            Focus();

        ColorResource accentColor = ColorResource.WHITE;
        if (HasFocus())
            accentColor = ResourceManager.ColorLoader.Get("highlight");
        Panel.AccentColor = accentColor;
        //DecreaseButton.Panel.AccentColor = accentColor;   // TODO
        //IncreaseButton.Panel.AccentColor = accentColor;

        DecreaseButton.Draw();
        IncreaseButton.Draw();

        bool decreaseClicked = DecreaseButton.IsClicked;
        bool increaseClicked = IncreaseButton.IsClicked;
        IsClicked = decreaseClicked || increaseClicked;

        if (IsClicked)
            Focus();

        if (decreaseClicked || (HasFocus() && Input.IsHotkeyActive("previous_subItem"))) {
            SelectedIndex = (SelectedIndex - 1 + Elements.Count) % Elements.Count;
            AudioManager.PlaySound("ui_interaction");
        } else if (increaseClicked || (HasFocus() && Input.IsHotkeyActive("next_subItem"))) {
            SelectedIndex = (SelectedIndex + 1) % Elements.Count;
            AudioManager.PlaySound("ui_interaction");
        }

        Panel.Draw();

        int textPosX = (int)(Bounds.x + Bounds.width / 2 - Raylib.MeasureText(SelectedElement.Text, FontSize) / 2);
        int textPosY = (int)(Bounds.y + Bounds.height / 2 - FontSize / 2);
        Raylib.DrawTextEx(Renderer.GuiFont.Resource, SelectedElement.Text, new Vector2(textPosX, textPosY), FontSize, FontSpacing, Raylib.WHITE);

    }

    internal record SelectionElement(string Text, object Element);
}
