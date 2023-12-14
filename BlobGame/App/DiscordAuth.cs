using System.Diagnostics;
using System.Dynamic;
using System.Net.Http.Json;
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Web;
using Microsoft.CSharp.RuntimeBinder;

namespace BlobGame.App;
/// <summary>
/// Class for managing discord authentication
/// </summary>
static internal class DiscordAuth{

    private const string DISCORD_API_ENDPOINT = "https://discord.com/api/v10";
    private const string SCOREBOARD_API_ENDPOINT = "https://robotino.ch/toasted/api";
    private const string USER_AGENT = "Toasted! (https://github.com/gartoks/BlobGame)";

    /// <summary>
    /// The current users access and refresh tokens
    /// </summary>
    private static Tokens? _Tokens;
    /// <summary>
    /// The current users username
    /// </summary>
    public static string Username { get; private set; } = "No user";
    /// <summary>
    /// The current users user id
    /// </summary>
    public static string UserID { get; private set; } = "No user";

    /// <summary>
    /// Retrieves the username and id for the current tokens.
    /// </summary>
    internal static async Task UpdateUserInfo(){
        if (!IsSignedIn){
            Username = "No user";
            UserID = "No user";
            return;
        }

        var client = new HttpClient();
        var request = new HttpRequestMessage();
        request.RequestUri = new Uri(DISCORD_API_ENDPOINT + "/users/@me");
        request.Method = HttpMethod.Get;
        request.Headers.Add("Accept", "*/*");
        request.Headers.Add("User-Agent", USER_AGENT);
        request.Headers.Add("Authorization", $"Bearer {_Tokens.AccessToken}");

        var response = await client.SendAsync(request);
        var result = await response.Content.ReadFromJsonAsync<JsonNode>();
        Username = result["global_name"].GetValue<string>();
        UserID = result["id"].GetValue<string>();
    }

    /// <summary>
    /// Indicates if a used is signed in
    /// </summary>
    internal static bool IsSignedIn { get => _Tokens != null && _Tokens.AccessToken != null && _Tokens.RefreshToken != null; }

    /// <summary>
    /// Sets the new tokens after testing their validity
    /// </summary>
    /// <param name="newTokens"></param>
    internal static async Task SetTokens(Tokens newTokens){
        if (await IsValidToken(newTokens)){
            _Tokens = newTokens;
        }
    }
    /// <summary>
    /// Returns the current tokens and refreshes them if needed.
    /// </summary>
    /// <returns></returns>
    internal static async Task<Tokens?> GetTokens(){
        if (_Tokens == null){
            return null;
        }
        if (await IsValidToken(_Tokens)){
            return _Tokens;
        }

        await RefreshTokens();
        return _Tokens;
    }

    /// <summary>
    /// Refreshes the access token.
    /// </summary>
    private static async Task RefreshTokens(){
        if (_Tokens == null){
            return;
        }

        var client = new HttpClient();
        var request = new HttpRequestMessage();
        request.RequestUri = new Uri(SCOREBOARD_API_ENDPOINT + "/game/refresh");
        request.Method = HttpMethod.Post;
        request.Headers.Add("Accept", "*/*");
        request.Headers.Add("User-Agent", USER_AGENT);
        request.Headers.Add("Authorization", $"Bearer {_Tokens.RefreshToken}");

        var response = await client.SendAsync(request);
        var new_token = await response.Content.ReadFromJsonAsync<JsonNode>();
        _Tokens = new Tokens(new_token["discord_access_token"].GetValue<string>(), _Tokens.RefreshToken);
    }

    /// <summary>
    /// Tests if the given tokens are still valid and not expired.
    /// </summary>
    /// <param name="tokens">The tokens to test</param>
    /// <returns></returns>
    private static async Task<bool> IsValidToken(Tokens tokens){
        if (tokens == null){
            return false;
        }
        
        var client = new HttpClient();
        var request = new HttpRequestMessage();
        request.RequestUri = new Uri(DISCORD_API_ENDPOINT + "/users/@me");
        request.Method = HttpMethod.Get;
        request.Headers.Add("Accept", "*/*");
        request.Headers.Add("User-Agent", USER_AGENT);
        request.Headers.Add("Authorization", $"Bearer {tokens.AccessToken}");

        var response = await client.SendAsync(request);
        return response.StatusCode == System.Net.HttpStatusCode.OK;
    }

    /// <summary>
    /// Signs in a user. This will open a browser and communicate with the scoreboard server.
    /// </summary>
    internal static async Task SignIn(){
        if (await IsValidToken(_Tokens)){
            return;
        }

        var client = new HttpClient();
        var request = new HttpRequestMessage();
        request.RequestUri = new Uri(SCOREBOARD_API_ENDPOINT + "/game/request_token");
        request.Method = HttpMethod.Get;

        request.Headers.Add("Accept", "*/*");
        request.Headers.Add("User-Agent", USER_AGENT);

        var response = await client.SendAsync(request);
        var result = await response.Content.ReadFromJsonAsync<JsonNode>();
        var exchange_token = result["exchange_token"].GetValue<string>();

        Process.Start(new ProcessStartInfo(){
            UseShellExecute = true,
            FileName = SCOREBOARD_API_ENDPOINT + "/game/auth?state=" + exchange_token,
        });

        Tokens tokens = null;

        while (tokens == null){
            await Task.Delay(1000);

            request = new HttpRequestMessage();
            request.RequestUri = new Uri(SCOREBOARD_API_ENDPOINT + "/game/request_token?exchange_token=" + exchange_token);
            request.Method = HttpMethod.Post;

            request.Headers.Add("Accept", "*/*");
            request.Headers.Add("User-Agent", USER_AGENT);

            response = await client.SendAsync(request);
            if (response.StatusCode == System.Net.HttpStatusCode.Accepted)
                continue;

            if (response.StatusCode == System.Net.HttpStatusCode.OK){
                var result2 = await response.Content.ReadFromJsonAsync<JsonNode>();
                tokens = new Tokens(result2["access_token"].GetValue<string>(), result2["refresh_token"].GetValue<string>());
            }
            break;
        }

        Debug.WriteLine(tokens);
        _Tokens = tokens;
    }

    /// <summary>
    /// Signs out the user. Also revokes the old tokens.
    /// </summary>
    internal static async Task SignOut(){
        if (_Tokens == null){
            return;
        }

        var client = new HttpClient();
        var request = new HttpRequestMessage();
        request.RequestUri = new Uri(SCOREBOARD_API_ENDPOINT + "/game/revoke");
        request.Method = HttpMethod.Post;
        request.Headers.Add("Accept", "*/*");
        request.Headers.Add("User-Agent", USER_AGENT);
        request.Headers.Add("Authorization", $"Bearer {_Tokens.AccessToken}");

        var response = await client.SendAsync(request);
        if (response.StatusCode == System.Net.HttpStatusCode.OK){
            _Tokens = null;
        }
        else{
            var ct = await response.Content.ReadAsStringAsync();
            Console.WriteLine(ct);
        }
    }

    public record Tokens(
        string AccessToken,
        string RefreshToken
    );
}
