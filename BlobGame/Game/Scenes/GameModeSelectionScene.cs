using BlobGame.Game.GameControllers;
using BlobGame.Game.GameModes;
using BlobGame.Game.Gui;
using BlobGame.ResourceHandling;
using System.Numerics;
using static BlobGame.Game.Gui.GuiSelector;

namespace BlobGame.Game.Scenes;
internal class GameModeSelectionScene : Scene {
    private GuiPanel BackgroundPanel { get; }

    private GuiTextButton BackButton { get; }
    private GuiTextButton PlayButton { get; }

    private GuiLabel GameModeLabel { get; }
    private GuiSelector GameModeSelector { get; }
    private GuiLabel GameControllerLabel { get; }
    private GuiSelector GameControllerSelector { get; }

    public GameModeSelectionScene() {
        BackgroundPanel = new GuiPanel(
                Application.BASE_WIDTH * 0.05f, Application.BASE_HEIGHT * 0.05f,
                Application.BASE_WIDTH * 0.9f, Application.BASE_HEIGHT * 0.8f,
                ResourceManager.GetColor("light_accent"),
                new Vector2(0, 0));

        BackButton = new GuiTextButton(
            Application.BASE_WIDTH * 0.05f, Application.BASE_HEIGHT * 0.95f,
            Application.BASE_WIDTH / 8f, Application.BASE_HEIGHT / 16f,
            "Back",
            new Vector2(0, 1));

        PlayButton = new GuiTextButton(
            Application.BASE_WIDTH * 0.95f, Application.BASE_HEIGHT * 0.95f,
            Application.BASE_WIDTH / 8f, Application.BASE_HEIGHT / 16f,
            "Play",
            new Vector2(1, 1));

        float xOffset = 0.1f;
        (GuiSelector gameModeSelector, GuiLabel gameModeLabel) = CreateGameSetting(
            "Game Mode", xOffset,
            IGameMode.GameModeTypes.Select(i => new SelectionElement(i.Key, i.Value)).ToArray(),
            0);
        GameModeLabel = gameModeLabel;
        GameModeSelector = gameModeSelector;
        xOffset += 0.1f;

        (GuiSelector gameControllerSelector, GuiLabel gameControllerLabel) = CreateGameSetting(
            "Game Controller", xOffset,
            IGameController.ControllerTypes.Select(i => new SelectionElement(i.Key, i.Value)).ToArray(),
            0);
        GameControllerLabel = gameControllerLabel;
        GameControllerSelector = gameControllerSelector;
        xOffset += 0.1f;
    }

    internal override void Load() {
    }

    internal override void Draw() {
        BackgroundPanel.Draw();

        GameModeLabel.Draw();
        GameControllerLabel.Draw();

        GameModeSelector.Draw();
        GameControllerSelector.Draw();

        if (BackButton.Draw())
            GameManager.SetScene(new MainMenuScene());
        if (PlayButton.Draw()) {
            IGameMode gameMode = IGameMode.CreateGameMode((Type)GameModeSelector.SelectedElement.Element, new Random().Next());
            IGameController controller = IGameController.CreateGameController((Type)GameControllerSelector.SelectedElement.Element);
            GameManager.SetScene(new GameScene(controller, gameMode));
        }

    }

    internal override void Unload() {
    }

    private (GuiSelector, GuiLabel) CreateGameSetting(string title, float xOffset, SelectionElement[] selectionElements, int selectedIndex) {
        GuiLabel label = new GuiLabel(
            Application.BASE_WIDTH * 0.05f, Application.BASE_HEIGHT * xOffset,
            Application.BASE_WIDTH / 4f, Application.BASE_HEIGHT / 16f,
            title,
            new Vector2(0, 0));

        GuiSelector selector = new GuiSelector(
            Application.BASE_WIDTH * 0.35f, Application.BASE_HEIGHT * xOffset,
            Application.BASE_WIDTH / 2f, Application.BASE_HEIGHT / 16f,
            selectionElements, selectedIndex < 0 ? 0 : selectedIndex,
            new Vector2(0, 0));

        return (selector, label);
    }
}
