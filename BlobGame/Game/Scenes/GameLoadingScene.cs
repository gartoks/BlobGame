using BlobGame.ResourceHandling;
using OpenTK.Mathematics;
using SimpleGL.Graphics.Rendering;
using SimpleGL.Util;
using System.Collections.Concurrent;

namespace BlobGame.Game.Scenes;
internal sealed class GameLoadingScene : Scene {

    private bool StartedLoading { get; set; }

    private ConcurrentDictionary<ResourceLoader, string[]> Items { get; set; }
    private int TotalItems { get; set; }
    private int LoadedItems { get; set; }

    public GameLoadingScene() {
        StartedLoading = false;

        Dictionary<ResourceLoader, string[]> items = new() {
            { ResourceManager.TextureLoader, new string[] {
                "title_logo",
                "melba_avatar",
                "rankup_arrow",
                "arena_bg",
                "marker",
                "dropper",
                "speechbubble",
                "lmb",
                "pointer",
                "blueberry_no_face",
                "strawberry_no_face",
                "0",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                "10",
                "0_shadow",
                "1_shadow",
                "2_shadow",
                "3_shadow",
                "4_shadow",
                "5_shadow",
                "6_shadow",
                "7_shadow",
                "8_shadow",
                "9_shadow",
                "10_shadow"
            } },
            { ResourceManager.ColorLoader, new string[] {
                "outline",
                "background",
                "highlight",
                "dark_accent",
                "light_accent"

            } },
            { ResourceManager.FontLoader, new string[] {
                "main",
                "gui",
                "Consolas"
            } },
            { ResourceManager.SoundLoader, new string[] {
                "ui_interaction",
                "piece_combination",
                "new_highscore",
                "game_loss"
            } },
            { ResourceManager.MusicLoader, new string[] {
                "crossinglike",
                "Melba_1",
                "Melba_2",
                "Melba_3",
                "Melba_s_Toasty_Game",
                "On_the_Surface",
                "synthyupdated"
            } },
            { ResourceManager.TextLoader, new string[] {
                "game_mode_descriptions",
                "scrollers"
            } },
            { ResourceManager.NPatchLoader, new string[] {
                "button_up",
                "button_selected",
                "panel",
            } },
        };
        Items = new(items);

        TotalItems = items.Select(kvp => kvp.Value.Length).Sum();
        LoadedItems = 0;
    }

    internal override void Load() {
        base.Load();

        // Load only gui needed stuff
        ResourceManager.ResourceLoaded += ResourceManager_ResourceLoaded;
    }

    internal override void Unload() {
        base.Unload();
        GameManager.LoadResources();
    }

    internal override void Render() {
        base.Render();
    }

    internal override void RenderGui() {
        base.RenderGui();

        const int X = 1000;
        const int EXTERNAL_WIDTH = 500;
        const int EXTERNAL_HEIGHT = 60;
        const int INTERNAL_WIDTH = 490;
        const int INTERNAL_HEIGHT = 50;

        Primitives.DrawRectangle(new Vector2(X, 400), new Vector2(EXTERNAL_WIDTH, EXTERNAL_HEIGHT), new Vector2(0, 0.5f), 0, 1, Color4.DarkGray);
        Primitives.DrawRectangle(new Vector2(X - 5, 400), new Vector2(INTERNAL_WIDTH * LoadedItems / (float)TotalItems, INTERNAL_HEIGHT), new Vector2(0, 0.5f), 0, 2, Color4.Lime);
    }

    internal override void Update(float dT) {
        base.Update(dT);

        if (!StartedLoading) {
            foreach (ResourceLoader loader in Items.Keys)
                loader.Load(Items[loader]);
            StartedLoading = true;
        }


        if (LoadedItems == TotalItems) {
            GameManager.SetScene(new MainMenuScene());
        }
    }

    private void ResourceManager_ResourceLoaded(string key, Type type) {
        LoadedItems++;

        if (LoadedItems == TotalItems)
            ResourceManager.ResourceLoaded -= ResourceManager_ResourceLoaded;

        Log.WriteLine($"Loaded {LoadedItems}/{TotalItems}");
    }
}
