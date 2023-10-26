﻿using BlobGame.Game.GUI;
using System.Numerics;

namespace BlobGame.Game.Scenes;
internal class SettingsScene : Scene {
    //Raylib.SetWindowSize(2560, 1440);
    //Raylib.SetWindowState(ConfigFlags.FLAG_WINDOW_UNDECORATED);

    internal override void Draw() {
        DrawBackButton();
    }

    private void DrawBackButton() {
        float w = Application.BASE_WIDTH / 8f;
        float h = Application.BASE_HEIGHT / 16f;

        float x = 180;
        float y = Application.BASE_HEIGHT - 150;
        bool pressed = GUITextButton.Draw(x, y, w, h, "Back", new Vector2(0, 1));

        if (pressed)
            GameManager.SetScene(new MainMenuScene());
    }
}
