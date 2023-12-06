namespace BlobGame.Util;
using System;
using System.Runtime.InteropServices;
static class ConsoleControl {
    const int SW_HIDE = 0;
    const int SW_SHOW = 5;

#if WINDOWS
    static readonly IntPtr handle = GetConsoleWindow();

    [DllImport("kernel32.dll")] static extern IntPtr GetConsoleWindow();
    [DllImport("user32.dll")] static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
#endif

    public static void Hide() {
#if WINDOWS
        ShowWindow(handle, SW_HIDE); //hide the console
#endif
    }
    public static void Show() {
#if WINDOWS
        ShowWindow(handle, SW_SHOW); //show the console
#endif
    }
}