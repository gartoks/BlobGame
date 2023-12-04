using BlobGame.App;
using BlobGame.Audio;
using BlobGame.ResourceHandling;
using BlobGame.Util;
using OpenTK.Mathematics;
using OpenTK.Windowing.GraphicsLibraryFramework;
using SimpleGL.Graphics.Rendering;

namespace BlobGame.Game.Gui;
internal class GuiSelector : InteractiveGuiElement {
    private const float BUTTON_SPACING = 10;

    private IReadOnlyList<SelectionElement> Elements { get; }

    public GuiNPatchPanel Panel { get; }
    public GuiTextButton DecreaseButton { get; }
    public GuiTextButton IncreaseButton { get; }

    private int FontSizeInt => (int)FontSize;
    private float FontSize { get; }
    //private float FontSpacing { get; }

    private int SelectedIndex { get; set; }
    public SelectionElement SelectedElement => Elements[SelectedIndex];

    public bool IsClicked { get; private set; }

    public GuiSelector(string boundsString, SelectionElement[] elements, int selectedIndex, int zIndex, Vector2? pivot = null)
        : this(GuiBoundsParser.Parse(boundsString), elements, selectedIndex, zIndex, pivot) {
    }

    private GuiSelector(Box2 bounds, SelectionElement[] elements, int selectedIndex, int zIndex, Vector2? pivot = null)
        : this(bounds.X(), bounds.Y(), bounds.Width(), bounds.Height(), elements, selectedIndex, zIndex, pivot) {
    }

    private GuiSelector(float x, float y, float w, float h, SelectionElement[] elements, int selectedIndex, int zIndex, Vector2? pivot = null)
        : base(x, y, w, h, zIndex, pivot) {
        float buttonSize = MathF.Min(w, h);

        Elements = elements;
        FontSize = buttonSize * 0.7f;
        //FontSpacing = FontSize / 64f;
        //Bounds = new Rectangle(x + buttonSize + 10, y, w - 2 * buttonSize - 20, h);

        Panel = new GuiNPatchPanel(Bounds.X() + buttonSize + BUTTON_SPACING, Bounds.Y(), Bounds.Width() - 2 * buttonSize - 2 * BUTTON_SPACING, Bounds.Height(), "button_up", zIndex, new Vector2(0, 0));
        DecreaseButton = new GuiTextButton(Bounds.X(), Bounds.Y(), buttonSize, buttonSize, "<", zIndex, new Vector2(0, 0));
        IncreaseButton = new GuiTextButton(Bounds.X() + Bounds.Width(), Bounds.Y(), buttonSize, buttonSize, ">", zIndex, new Vector2(1, 0));

        SelectedIndex = selectedIndex;
    }

    internal override void Load() {
        base.Load();

        Panel.Load();
        DecreaseButton.Load();
        IncreaseButton.Load();
    }

    protected override void DrawInternal() {
        bool shouldFocus = IsHovered && Input.IsMouseButtonActive(MouseButton.Left);
        if (shouldFocus)
            Focus();

        string texture = "button_up";
        if (HasFocus())
            texture = "button_selected";
        Panel.TextureKey = texture;

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

        MeshFont font = Fonts.GetGuiFont(FontSizeInt);
        Vector2 textSize = font.MeasureText(SelectedElement.Text);

        int textPosX = (int)(Bounds.X() + Bounds.Width() / 2 - textSize.X / 2);
        int textPosY = (int)(Bounds.Y() + Bounds.Height() / 2 - FontSize / 2);
        Primitives.DrawText(font, SelectedElement.Text, Color4.White, new Vector2(textPosX, textPosY), Vector2.Zero, 0, ZIndex + 1);
    }

    internal record SelectionElement(string Text, object Element);
}
