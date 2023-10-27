using BlobGame.ResourceHandling;
using Raylib_CsLo;
using System.Numerics;

namespace BlobGame.Game.Gui;
internal class GuiSelector {
    private IReadOnlyList<SelectionElement> Elements { get; }

    private int FontSize { get; }
    private Rectangle Bounds { get; }

    private GUIPanel Panel { get; }
    private GUITextButton DecreaseButton { get; }
    private GUITextButton IncreaseButton { get; }

    private int SelectedIndex { get; set; }
    public SelectionElement SelectedElement => Elements[SelectedIndex];

    public GuiSelector(Vector2 pos, Vector2 size, SelectionElement[] elements, int selectedIndex, Vector2? pivot = null)
        : this(pos.X, pos.Y, size.X, size.Y, elements, selectedIndex, pivot) {
    }

    public GuiSelector(float x, float y, float w, float h, SelectionElement[] elements, int selectedIndex, Vector2? pivot = null) {
        if (pivot != null) {
            x += -w * pivot.Value.X;
            y += -h * pivot.Value.Y;
        }
        float buttonSize = MathF.Min(w, h);

        Elements = elements;
        FontSize = (int)(buttonSize * 0.7f);
        Bounds = new Rectangle(x + buttonSize + 10, y, w - 2 * buttonSize - 20, h);

        Panel = new GUIPanel(x + buttonSize + 10, y, w - 2 * buttonSize - 20, h, ResourceManager.GetColor("light_accent"), new Vector2(0, 0));
        DecreaseButton = new GUITextButton(x, y, buttonSize, buttonSize, "<", new Vector2(0, 0));
        IncreaseButton = new GUITextButton(x + w, y, buttonSize, buttonSize, ">", new Vector2(1, 0));

        SelectedIndex = selectedIndex;
    }


    internal bool Draw() {
        int textPosX = (int)(Bounds.x + Bounds.width / 2 - Raylib.MeasureText(SelectedElement.Text, FontSize) / 2);
        int textPosY = (int)(Bounds.y + Bounds.height / 2 - FontSize / 2);

        bool decreaseClicked = DecreaseButton.Draw();
        bool increaseClicked = IncreaseButton.Draw();

        if (decreaseClicked)
            SelectedIndex = (SelectedIndex - 1 + Elements.Count) % Elements.Count;
        else if (increaseClicked)
            SelectedIndex = (SelectedIndex + 1) % Elements.Count;

        Panel.Draw();
        Raylib.DrawText(SelectedElement.Text, textPosX, textPosY, FontSize, Raylib.WHITE);


        return decreaseClicked || increaseClicked;
    }

    internal record SelectionElement(string Text, object Element);
}
