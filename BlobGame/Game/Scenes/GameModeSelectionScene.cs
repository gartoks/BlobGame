using BlobGame.Game.GameControllers;
using BlobGame.Game.GameModes;
using BlobGame.Game.Gui;
using BlobGame.ResourceHandling;
using BlobGame.ResourceHandling.Resources;
using System.Numerics;
using static BlobGame.Game.Gui.GuiSelector;

namespace BlobGame.Game.Scenes;
internal class GameModeSelectionScene : Scene {
    private TextResource GameModeDescriptionsText { get; set; }

    private GuiPanel BackgroundPanel { get; }

    private GuiLabel GameModeLabel { get; }
    private GuiSelector GameModeSelector { get; }
    private GuiLabel GameControllerLabel { get; }
    private GuiSelector GameControllerSelector { get; }

    private GuiDynamicLabel GameModeDescriptionLabel { get; }

    private GuiLabel PortLabel { get; }
    private GuiTextbox PortTextBox { get; }

    private GuiTextButton PlayButton { get; }
    private GuiTextButton BackButton { get; }

    public GameModeSelectionScene() {
        BackgroundPanel = new GuiPanel("0.05 0.05 0.9 0.8", new Vector2(0, 0));

        BackButton = new GuiTextButton("0.05 0.95 0.125 0.0625", "Back", new Vector2(0, 1));
        PlayButton = new GuiTextButton("0.95 0.95 0.125 0.0625", "Play", new Vector2(1, 1));

        float yOffset = 0.1f;
        (GuiSelector gameModeSelector, GuiLabel gameModeLabel) = CreateSelectionElement(
            "Game Mode", yOffset,
            IGameMode.GameModeTypes.Select(i => new SelectionElement(i.Key, i.Value)).ToArray(),
            0);
        GameModeLabel = gameModeLabel;
        GameModeSelector = gameModeSelector;
        yOffset += 0.1f;

        (GuiSelector gameControllerSelector, GuiLabel gameControllerLabel) = CreateSelectionElement(
            "Game Controller", yOffset,
            IGameController.ControllerTypes.Select(i => new SelectionElement(i.Key, i.Value)).ToArray(),
            0);
        GameControllerLabel = gameControllerLabel;
        GameControllerSelector = gameControllerSelector;
        yOffset += 0.1f;

        PortLabel = new GuiLabel($"0.05 {yOffset} 0.25 0.0625", "Port", new Vector2(0, 0));
        PortLabel.Enabled = false;
        PortTextBox = new GuiTextbox($"0.35 {yOffset} 0.125 0.0625", new Vector2(0, 0.5f)) {
            CharFilter = char.IsDigit
        };
        PortTextBox.Text = "1337";
        PortTextBox.Enabled = false;
        yOffset += 0.1f;

        GameModeDescriptionLabel = new GuiDynamicLabel(Application.BASE_WIDTH * 0.1f, (yOffset + 0.025f) * Application.BASE_HEIGHT, string.Empty, 60);
    }

    internal override void Load() {
        GameModeDescriptionsText = ResourceManager.TextLoader.Get("game_mode_descriptions");

        LoadAllGuiElements();
    }

    internal override void Draw() {
        BackgroundPanel.Draw();

        GameModeLabel.Draw();
        GameControllerLabel.Draw();

        GameModeSelector.Draw();
        GameControllerSelector.Draw();

        GameModeDescriptionLabel.Text = GameModeDescriptionsText.Resource[GameModeSelector.SelectedElement.Text];
        GameModeDescriptionLabel.Draw();

        PortLabel.Draw();
        PortTextBox.Enabled = (Type)GameControllerSelector.SelectedElement.Element == typeof(SocketController);
        PortLabel.Enabled = PortTextBox.Enabled;
        PortTextBox.Draw();

        BackButton.Draw();
        PlayButton.Draw();

        if (BackButton.IsClicked)
            GameManager.SetScene(new MainMenuScene());
        if (PlayButton.IsClicked) {
            IGameMode gameMode = IGameMode.CreateGameMode((Type)GameModeSelector.SelectedElement.Element, new Random().Next());

            IGameController controller;
            if ((Type)GameControllerSelector.SelectedElement.Element == typeof(SocketController))
                controller = IGameController.CreateGameController((Type)GameControllerSelector.SelectedElement.Element, 0, int.Parse(PortTextBox.Text));
            else
                controller = IGameController.CreateGameController((Type)GameControllerSelector.SelectedElement.Element);

            GameManager.SetScene(new GameScene(controller, gameMode));
        }
    }

    internal override void Unload() {
    }

    private (GuiSelector, GuiLabel) CreateSelectionElement(string title, float yOffset, SelectionElement[] selectionElements, int selectedIndex) {
        GuiLabel label = new GuiLabel($"0.05 {yOffset} 0.25 0.0625", title, new Vector2(0, 0));
        label.TextAlignment = eTextAlignment.Center;

        GuiSelector selector = new GuiSelector($"0.35 {yOffset} 0.5 {1f / 16f}",
            selectionElements, selectedIndex < 0 ? 0 : selectedIndex,
            new Vector2(0, 0));

        return (selector, label);
    }
}
