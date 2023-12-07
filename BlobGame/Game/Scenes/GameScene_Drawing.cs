using BlobGame.Drawing;
using BlobGame.Game.Blobs;
using BlobGame.Game.GameModes;
using BlobGame.Game.Gui;
using BlobGame.ResourceHandling;
using BlobGame.ResourceHandling.Resources;
using BlobGame.Util;
using Raylib_CsLo;
using System.Numerics;

namespace BlobGame.Game.Scenes;
internal sealed partial class GameScene : Scene {
    private void DrawMenuButton() {
        MenuButton.Draw();

        if (MenuButton.IsClicked) {
            IsMenuOpen = true;
        }
    }

    private void DrawMenu() {
        MenuPanel.Draw();
        ToMainMenuButton.Draw();
        ContinueButton.Draw();
        MenuLabel.Draw();

        if (ContinueButton.IsClicked)
            IsMenuOpen = false;
        if (ToMainMenuButton.IsClicked)
            GameManager.SetScene(new MainMenuScene());
    }

    private void DrawSplats(float dT) {
        foreach (KeyValuePair<Guid, (TextureResource tex, float alpha, float rotation, Vector2 position)> splat in Splats.ToList()) {
            Guid id = splat.Key;
            TextureResource texture = splat.Value.tex;
            float alpha = splat.Value.alpha;
            float rotation = splat.Value.rotation;
            Vector2 position = splat.Value.position;

            texture.Draw(new Rectangle(position.X, position.Y, 200, 200), new Vector2(0.5f, 0.5f), rotation, Raylib.WHITE.ChangeAlpha((int)(255 * alpha)));

            float newAlpha = alpha - 2 * dT;

            if (newAlpha <= 0) {
                Splats.Remove(id);
            } else {
                Splats[id] = (texture, newAlpha, rotation, position);
            }
        }
    }

    private void DrawRankupChart() {
        //const float size = 75;
        //const float radius = 200;

        //float ruaW = RankupArrowTexture.Resource.width;
        //float ruaH = RankupArrowTexture.Resource.height;

        float cX = 1620;
        float cY = 735f;

        RankupArrowTexture.Draw(new Rectangle(cX, cY, 520, 520), new Vector2(0.5f, 0.5f));

        /*Raylib.DrawTexturePro(
            RankupArrowTexture.Resource,
            new Rectangle(0, 0, ruaW, ruaH),
            new Rectangle(cX, cY, 521, 521),
            new Vector2(ruaW * 1.3f / 2, ruaH * 1.3f / 2),
            0,
            new Color(255, 255, 255, 255));*/
        /*
                int i = 0;
                foreach (BlobData blobType in Game.Blobs.Values) {
                    float angle = (i + 1) / (float)(Game.Blobs.Count + 1) * MathF.Tau - MathF.PI / 2f;

                    float x = cX + radius * MathF.Cos(angle);
                    float y = cY + radius * MathF.Sin(angle);

                    string texKey = Game.Blobs[blobType.Id].Name;
                    Texture tex = ResourceManager.TextureLoader.Get($"{texKey}_shadow").Resource;
                    float w = tex.width;
                    float h = tex.height;

                    Raylib.DrawTexturePro(
                        tex,
                        new Rectangle(0, 0, w, h),
                        new Rectangle(x, y, size, size),
                        new Vector2(size / 2, size / 2), 0, Raylib.WHITE);
                    i++;
                }*/
    }

    private void DrawArenaBackground() {
        ArenaBackgroundTexture.Draw(new Rectangle(0, 0, 711, 863), new Vector2(0.5f, 0));
    }

    private void DrawArenaBox() {
        ArenaBoxTexture.Draw(new Rectangle(0, 0, 711, 863), new Vector2(0.5f, 0));
    }

    private void DrawNextBlob() {
        // Hightlight
        MarkerTexture.Draw(new Vector2(IGameMode.ARENA_WIDTH * 0.75f, 0), new Vector2(0.5f, 0.5f), null, 0, ResourceManager.ColorLoader.Get("light_accent").Resource);

        BlobData blob = Game.Blobs[Game.NextBlob];
        TextureResource tex = ResourceManager.TextureLoader.Get(blob.Name);

        // Blob
        tex.Draw(
            new Vector2(IGameMode.ARENA_WIDTH * 0.75f, 0),
            new Vector2(0.5f, 0.5f),
            new Vector2(0.25f, 0.25f));

        Vector2 textPos = new Vector2(IGameMode.ARENA_WIDTH * 0.75f, -310);
        Raylib.DrawTextPro(
            Renderer.MainFont.Resource,
            "NEXT",
            textPos,
            textPos / 2f,
            -25.5f,
            80, 5, ResourceManager.ColorLoader.Get("font_dark").Resource);
    }

    private void DrawHeldBlob() {
        // Hightlight
        MarkerTexture.Draw(new Vector2(IGameMode.ARENA_WIDTH * -0.75f, 0), new Vector2(0.5f, 0.5f), null, 0, ResourceManager.ColorLoader.Get("light_accent").Resource);

        if (Game.HeldBlob != -1) {
            BlobData blob = Game.Blobs[Game.HeldBlob];
            TextureResource tex = ResourceManager.TextureLoader.Get(blob.Name);

            // Blob
            tex.Draw(
                new Vector2(IGameMode.ARENA_WIDTH * -0.75f, 0),
                new Vector2(0.5f, 0.5f),
                new Vector2(0.25f, 0.25f));
        }

        Vector2 textPos = new Vector2(IGameMode.ARENA_WIDTH * -1.85f, -280);
        Raylib.DrawTextPro(
            Renderer.MainFont.Resource,
            "HELD",
            textPos,
            textPos / 2f,
            0,
            80, 5, ResourceManager.ColorLoader.Get("font_dark").Resource);
    }

    private void DrawDropper(float x) {
        DropperTexture.Draw(
            new Rectangle(x, 3f * IGameMode.ARENA_SPAWN_Y_OFFSET, 220, 150),
            new Vector2(0.5f, 0.5f));
    }

    private void DrawDropIndicator(float x) {
        Raylib.DrawRectanglePro(
            new Rectangle(x, 0, DROP_INDICATOR_WIDTH, IGameMode.ARENA_HEIGHT),
            new Vector2(DROP_INDICATOR_WIDTH / 2f, 0),
            0,
            ResourceManager.ColorLoader.Get("background").Resource.ChangeAlpha(128));
    }

    private void DrawCurrentBlob(float x) {
        if (!Game.CanSpawnBlob)
            return;

        BlobData blob = Game.Blobs[Game.CurrentBlob];
        TextureResource tex = ResourceManager.TextureLoader.Get(blob.Name);

        tex.Draw(
            new Vector2(x, IGameMode.ARENA_SPAWN_Y_OFFSET),
            new Vector2(0.5f, 0.5f),
            new Vector2(blob.TextureScale.X, blob.TextureScale.Y),
            Game.SpawnRotation * RayMath.RAD2DEG);
    }

    private void DrawScoreboard() {
        const float x = 122.5f;
        const float y = 350;
        const float w = 400;

        DrawCurrentScore(x, y - 100, w);
        DrawHighscores(x, y + 120, w);
    }

    private void DrawCurrentScore(float x, float y, float w) {
        CurrentScoreBackgroundTexture.Draw(new Rectangle(x - 1.45f * w, y, 2.5f * w, 2.5f * w / CurrentScoreBackgroundTexture.Resource.width * CurrentScoreBackgroundTexture.Resource.height), new Vector2(0, 0.3f));
        CurrentScoreBitmapFont.DrawAsBitmapFont(Game.Score.ToString(), 10, 120, new Vector2(x + w * 0.85f, y), new Vector2(1, 0));
    }

    private void DrawHighscores(float x, float y, float w) {
        ScoresBackgroundTexture.Draw(
            new Rectangle(x - w * 0.4f, y - 7, w * 1.6f, w * 1.65f));

        DrawScoreValue(x, y, w, GameManager.Scoreboard.GetGlobalHighscore(Game), "font_dark");

        for (int i = 0; i < GameManager.Scoreboard.GetDailyHighscores(Game).Count; i++) {
            DrawScoreValue(x, y + 160 + 130 * i, w, GameManager.Scoreboard.GetDailyHighscores(Game)[i]);
        }
    }

    private void DrawScoreValue(float x, float y, float w, int score, string colorKey = null, bool useMainFont = false, float fontSize = 90) {
        if (colorKey == null)
            colorKey = "font_light";

        FontResource font = useMainFont ? Renderer.MainFont : Renderer.GuiFont;

        string scoreStr = $"{score}";
        Vector2 scoreTextSize = Raylib.MeasureTextEx(font.Resource, scoreStr, 100, 10);
        Raylib.DrawTextPro(
                font.Resource,
                scoreStr,
                new Vector2(x + w - 50 - scoreTextSize.X, y + 5),
                new Vector2(scoreTextSize.Y / 2f, 0),
                0,
                fontSize,
                10,
                ResourceManager.ColorLoader.Get(colorKey).Resource);
    }

    private void DrawGameOverScreen() {
        GameOverPanel.Draw();
        GameOverTexture.Draw(new Rectangle(
            Application.BASE_WIDTH / 2f,
            Application.BASE_HEIGHT * 0.225f,
            150 / (float)GameOverTexture.Resource.height * GameOverTexture.Resource.width,
            150),
            new Vector2(0.5f, 0.5f));

        RetryButton.Draw();
        ToMainMenuButton.Draw();

        bool isNewHighscore = Game.Score > GameManager.Scoreboard.GetGlobalHighscore(Game);
        GuiLabel scoreLabel = new GuiLabel("0.5 0.375 1100px 240px",
            $"{Game.Score}",
            new Vector2(0.5f, 0));
        scoreLabel.Color = ResourceManager.ColorLoader.Get("font_dark");
        scoreLabel.Draw();

        if (isNewHighscore) {
            GuiLabel highScoreLabel = new GuiLabel("0.5 0.43 1100px 100px",
            $"New Highscore!",
            new Vector2(0.5f, 1));
            highScoreLabel.Color = ResourceManager.ColorLoader.Get("font_dark");
            highScoreLabel.Draw();
        }

        if (RetryButton.IsClicked)
            GameManager.SetScene(new GameScene(Controller, IGameMode.CreateGameMode(Game.GetType(), new Random().Next())));
        if (ToMainMenuButton.IsClicked)
            GameManager.SetScene(new MainMenuScene());
    }
}
