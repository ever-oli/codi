/**
 * API Discovery Extension
 *
 * Provides commands to search and discover free public APIs from the
 * public-apis database (github.com/public-apis/public-apis).
 *
 * Usage:
 *   pi --extension api-discovery.ts
 *
 * Commands:
 *   /api search <query>     - Search for APIs by keyword
 *   /api category <name>    - List APIs in a category
 *   /api categories         - Show all categories
 *   /api random             - Get a random API to explore
 *   /api use <name>         - Generate code snippet for an API
 */

import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";

// ============================================================================
// API Database (curated from public-apis/public-apis)
// ============================================================================

interface ApiEntry {
	name: string;
	description: string;
	category: string;
	url: string;
	auth: "No" | "apiKey" | "OAuth" | "X-Mashape-Key";
	https: boolean;
	cors: "Yes" | "No" | "Unknown";
	exampleEndpoint?: string;
}

// Curated list of the most useful free APIs
const API_DATABASE: ApiEntry[] = [
	// Finance & Crypto
	{ name: "CoinGecko", description: "Crypto prices, market data, exchanges", category: "Finance", url: "https://www.coingecko.com/en/api", auth: "No", https: true, cors: "Yes", exampleEndpoint: "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd" },
	{ name: "ExchangeRate-API", description: "Currency exchange rates", category: "Finance", url: "https://www.exchangerate-api.com/", auth: "No", https: true, cors: "Yes", exampleEndpoint: "https://open.er-api.com/v6/latest/USD" },
	{ name: "Alpha Vantage", description: "Stock prices and forex", category: "Finance", url: "https://www.alphavantage.co/", auth: "apiKey", https: true, cors: "No", exampleEndpoint: "https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=IBM&apikey=DEMO" },
	{ name: "FRED", description: "Federal Reserve economic data", category: "Finance", url: "https://fred.stlouisfed.org/", auth: "apiKey", https: true, cors: "No" },
	{ name: "CoinCap", description: "Real-time crypto prices", category: "Finance", url: "https://docs.coincap.io/", auth: "No", https: true, cors: "Yes", exampleEndpoint: "https://api.coincap.io/v2/assets/bitcoin" },
	{ name: "CoinDesk", description: "Bitcoin price index", category: "Finance", url: "https://www.coindesk.com/coindesk-api", auth: "No", https: true, cors: "Unknown", exampleEndpoint: "https://api.coindesk.com/v1/bpi/currentprice.json" },
	
	// Weather
	{ name: "Open-Meteo", description: "Weather forecasts, no API key", category: "Weather", url: "https://open-meteo.com/", auth: "No", https: true, cors: "Yes", exampleEndpoint: "https://api.open-meteo.com/v1/forecast?latitude=52.52&longitude=13.41&current_weather=true" },
	{ name: "wttr.in", description: "Simple weather in terminal/JSON", category: "Weather", url: "https://wttr.in/", auth: "No", https: true, cors: "Yes", exampleEndpoint: "https://wttr.in/London?format=j1" },
	{ name: "WeatherAPI", description: "Weather data and forecasts", category: "Weather", url: "https://www.weatherapi.com/", auth: "apiKey", https: true, cors: "No" },
	{ name: "Visual Crossing", description: "Weather history and forecast", category: "Weather", url: "https://www.visualcrossing.com/weather-api", auth: "apiKey", https: true, cors: "Yes" },
	
	// News
	{ name: "GNews", description: "News headlines from Google", category: "News", url: "https://gnews.io/", auth: "apiKey", https: true, cors: "Yes", exampleEndpoint: "https://gnews.io/api/v4/top-headlines?token=API_KEY&lang=en" },
	{ name: "NewsAPI", description: "Breaking news from sources", category: "News", url: "https://newsapi.org/", auth: "apiKey", https: true, cors: "Yes" },
	{ name: "Currents API", description: "Latest news worldwide", category: "News", url: "https://currentsapi.services/", auth: "apiKey", https: true, cors: "Yes" },
	{ name: "HackerNews", description: "Hacker News stories", category: "News", url: "https://github.com/HackerNews/API", auth: "No", https: true, cors: "Yes", exampleEndpoint: "https://hacker-news.firebaseio.com/v0/topstories.json" },
	
	// Geolocation
	{ name: "ip-api", description: "IP geolocation lookup", category: "Geolocation", url: "http://ip-api.com/", auth: "No", https: false, cors: "Yes", exampleEndpoint: "http://ip-api.com/json/8.8.8.8" },
	{ name: "Nominatim", description: "OpenStreetMap geocoding", category: "Geolocation", url: "https://nominatim.org/", auth: "No", https: true, cors: "Yes", exampleEndpoint: "https://nominatim.openstreetmap.org/search?q=New+York&format=json" },
	{ name: "REST Countries", description: "Country information", category: "Geolocation", url: "https://restcountries.com/", auth: "No", https: true, cors: "Yes", exampleEndpoint: "https://restcountries.com/v3.1/name/germany" },
	{ name: "ZipCodeAPI", description: "Zip code lookup", category: "Geolocation", url: "https://www.zipcodeapi.com/", auth: "apiKey", https: true, cors: "Yes" },
	{ name: "GeoJS", description: "IP geolocation", category: "Geolocation", url: "https://www.geojs.io/", auth: "No", https: true, cors: "Yes", exampleEndpoint: "https://get.geojs.io/v1/ip/geo.json" },
	
	// Developer Tools
	{ name: "GitHub", description: "GitHub REST API", category: "Development", url: "https://docs.github.com/en/rest", auth: "apiKey", https: true, cors: "Yes", exampleEndpoint: "https://api.github.com/repos/public-apis/public-apis" },
	{ name: "JSONPlaceholder", description: "Fake REST API for testing", category: "Development", url: "https://jsonplaceholder.typicode.com/", auth: "No", https: true, cors: "Yes", exampleEndpoint: "https://jsonplaceholder.typicode.com/posts" },
	{ name: "httpbin", description: "HTTP request & response service", category: "Development", url: "https://httpbin.org/", auth: "No", https: true, cors: "Yes", exampleEndpoint: "https://httpbin.org/get" },
	{ name: "RandomUser", description: "Generate random user data", category: "Development", url: "https://randomuser.me/", auth: "No", https: true, cors: "Yes", exampleEndpoint: "https://randomuser.me/api/" },
	{ name: "Public APIs List", description: "Directory of public APIs", category: "Development", url: "https://github.com/public-apis/public-apis", auth: "No", https: true, cors: "Yes" },
	
	// Science & Data
	{ name: "NASA APIs", description: "Space and satellite data", category: "Science", url: "https://api.nasa.gov/", auth: "apiKey", https: true, cors: "Yes", exampleEndpoint: "https://api.nasa.gov/planetary/apod?api_key=DEMO_KEY" },
	{ name: "Open Library", description: "Book database", category: "Books", url: "https://openlibrary.org/developers/api", auth: "No", https: true, cors: "Yes", exampleEndpoint: "https://openlibrary.org/api/books?bibkeys=ISBN:0451526538&format=json" },
	{ name: "Wikidata", description: "Structured knowledge base", category: "Open Data", url: "https://www.wikidata.org/w/api.php", auth: "No", https: true, cors: "Yes" },
	{ name: "Wikipedia", description: "Wikipedia API", category: "Open Data", url: "https://www.mediawiki.org/wiki/API:Main_page", auth: "No", https: true, cors: "Yes", exampleEndpoint: "https://en.wikipedia.org/api/rest_v1/page/summary/Artificial_intelligence" },
	
	// Entertainment
	{ name: "PokéAPI", description: "Pokémon data", category: "Games", url: "https://pokeapi.co/", auth: "No", https: true, cors: "Yes", exampleEndpoint: "https://pokeapi.co/api/v2/pokemon/pikachu" },
	{ name: "Deck of Cards", description: "Card deck simulator", category: "Games", url: "https://deckofcardsapi.com/", auth: "No", https: true, cors: "Yes", exampleEndpoint: "https://deckofcardsapi.com/api/deck/new/shuffle/" },
	{ name: "TriviaDB", description: "Trivia questions", category: "Games", url: "https://opentdb.com/", auth: "No", https: true, cors: "Yes", exampleEndpoint: "https://opentdb.com/api.php?amount=10" },
	{ name: "Jikan", description: "MyAnimeList unofficial API", category: "Anime", url: "https://jikan.moe/", auth: "No", https: true, cors: "Yes", exampleEndpoint: "https://api.jikan.moe/v4/anime/1" },
	
	// Social & Communication
	{ name: "QR Code Generator", description: "Generate QR codes", category: "Images", url: "https://goqr.me/api/", auth: "No", https: true, cors: "Yes", exampleEndpoint: "https://api.qrserver.com/v1/create-qr-code/?data=Hello&size=200x200" },
	{ name: "Unsplash", description: "Free stock photos", category: "Photography", url: "https://unsplash.com/developers", auth: "apiKey", https: true, cors: "Yes" },
	{ name: "Pexels", description: "Free stock photos and videos", category: "Photography", url: "https://www.pexels.com/api/", auth: "apiKey", https: true, cors: "Yes" },
	{ name: "Lorem Picsum", description: "Random placeholder images", category: "Images", url: "https://picsum.photos/", auth: "No", https: true, cors: "Yes", exampleEndpoint: "https://picsum.photos/200/300" },
	
	// Health
	{ name: "COVID-19", description: "COVID-19 statistics", category: "Health", url: "https://disease.sh/", auth: "No", https: true, cors: "Yes", exampleEndpoint: "https://disease.sh/v3/covid-19/all" },
	
	// Environment
	{ name: "OpenUV", description: "UV index data", category: "Environment", url: "https://www.openuv.io/", auth: "apiKey", https: true, cors: "Unknown" },
	{ name: "AirVisual", description: "Air quality data", category: "Environment", url: "https://www.iqair.com/air-pollution-data-api", auth: "apiKey", https: true, cors: "Unknown" },
	
	// Transportation
	{ name: "Transport for London", description: "London transit data", category: "Transportation", url: "https://api-portal.tfl.gov.uk/", auth: "apiKey", https: true, cors: "Yes" },
	{ name: "ADS-B Exchange", description: "Aircraft tracking", category: "Transportation", url: "https://www.adsbexchange.com/", auth: "apiKey", https: true, cors: "Unknown" },
];

// ============================================================================
// Categories
// ============================================================================

const CATEGORIES = [...new Set(API_DATABASE.map(a => a.category))].sort();

// ============================================================================
// Code Generation
// ============================================================================

function generateCodeSnippet(api: ApiEntry, language: "typescript" | "python" | "curl"): string {
	const endpoint = api.exampleEndpoint || api.url;
	
	switch (language) {
		case "typescript":
			if (api.auth === "No") {
				return `// ${api.name} - ${api.description}
const response = await fetch('${endpoint}');
const data = await response.json();
console.log(data);`;
			} else {
				return `// ${api.name} - ${api.description}
// Requires API key from ${api.url}
const API_KEY = process.env.${api.name.toUpperCase().replace(/[^A-Z]/g, '_')}_API_KEY;
const response = await fetch('${endpoint.includes('apikey') ? endpoint.replace('apikey=DEMO', 'apikey=API_KEY') : endpoint + '&apikey=' + '${API_KEY}'}', {
  headers: { 'Authorization': \`Bearer \${API_KEY}\` }
});
const data = await response.json();
console.log(data);`;
			}
		
		case "python":
			if (api.auth === "No") {
				return `# ${api.name} - ${api.description}
import requests

response = requests.get('${endpoint}')
data = response.json()
print(data)`;
			} else {
				return `# ${api.name} - ${api.description}
# Requires API key from ${api.url}
import requests
import os

API_KEY = os.environ.get('${api.name.toUpperCase().replace(/[^A-Z]/g, '_')}_API_KEY')
response = requests.get('${endpoint}', headers={'Authorization': f'Bearer {API_KEY}'})
data = response.json()
print(data)`;
			}
		
		case "curl":
			if (api.auth === "No") {
				return `# ${api.name} - ${api.description}
curl -s '${endpoint}' | jq .`;
			} else {
				return `# ${api.name} - ${api.description}
# Requires API key from ${api.url}
curl -s '${endpoint}' \\
  -H "Authorization: Bearer $API_KEY" | jq .`;
			}
	}
}

// ============================================================================
// Extension Registration
// ============================================================================

export default function apiDiscoveryExtension(pi: ExtensionAPI) {
	pi.registerCommand("api", {
		description: "Search and discover free public APIs",
		getArgumentCompletions: (prefix) => {
			const subcommands = ["search", "category", "categories", "random", "use"];
			return subcommands
				.filter(s => s.startsWith(prefix))
				.map(s => ({ value: s, label: s }));
		},
		handler: async (args, ctx) => {
			const parts = args.trim().split(/\s+/);
			const subcommand = parts[0] || "categories";
			
			switch (subcommand) {
				case "search": {
					const query = parts.slice(1).join(" ").toLowerCase();
					if (!query) {
						ctx.ui.notify("Usage: /api search <query>", "warning");
						return;
					}
					
					const results = API_DATABASE.filter(api => 
						api.name.toLowerCase().includes(query) ||
						api.description.toLowerCase().includes(query) ||
						api.category.toLowerCase().includes(query)
					);
					
					if (results.length === 0) {
						ctx.ui.notify(`No APIs found for "${query}"`, "info");
						return;
					}
					
					const lines = [
						`=== API Search: "${query}" (${results.length} results) ===`,
						"",
						...results.map(api => {
							const auth = api.auth === "No" ? "🆓" : "🔑";
							return `${auth} ${api.name} [${api.category}]\n   ${api.description}\n   ${api.url}`;
						}),
					];
					
					ctx.ui.showPanel("API Search Results", lines.join("\n"));
					break;
				}
				
				case "category": {
					const cat = parts.slice(1).join(" ");
					if (!cat) {
						ctx.ui.notify("Usage: /api category <name>", "warning");
						ctx.ui.notify(`Categories: ${CATEGORIES.join(", ")}`, "info");
						return;
					}
					
					const results = API_DATABASE.filter(api => 
						api.category.toLowerCase() === cat.toLowerCase()
					);
					
					if (results.length === 0) {
						ctx.ui.notify(`No APIs in category "${cat}". Categories: ${CATEGORIES.join(", ")}`, "info");
						return;
					}
					
					const lines = [
						`=== ${cat} APIs (${results.length}) ===`,
						"",
						...results.map(api => {
							const auth = api.auth === "No" ? "🆓" : "🔑";
							return `${auth} ${api.name}\n   ${api.description}\n   ${api.exampleEndpoint || api.url}`;
						}),
					];
					
					ctx.ui.showPanel(`${cat} APIs`, lines.join("\n"));
					break;
				}
				
				case "categories": {
					const lines = [
						"=== API Categories ===",
						"",
						...CATEGORIES.map(cat => {
							const count = API_DATABASE.filter(a => a.category === cat).length;
							const free = API_DATABASE.filter(a => a.category === cat && a.auth === "No").length;
							return `${cat}: ${count} APIs (${free} free/no-auth)`;
						}),
						"",
						"Use /api category <name> to list APIs in a category",
					];
					
					ctx.ui.showPanel("API Categories", lines.join("\n"));
					break;
				}
				
				case "random": {
					const api = API_DATABASE[Math.floor(Math.random() * API_DATABASE.length)];
					const auth = api.auth === "No" ? "🆓 No auth required" : `🔑 ${api.auth}`;
					
					const lines = [
						`=== Random API: ${api.name} ===`,
						"",
						`Category: ${api.category}`,
						`Description: ${api.description}`,
						`Auth: ${auth}`,
						`HTTPS: ${api.https ? "Yes" : "No"}`,
						`CORS: ${api.cors}`,
						`URL: ${api.url}`,
					];
					
					if (api.exampleEndpoint) {
						lines.push("", `Example: ${api.exampleEndpoint}`);
					}
					
					lines.push("", "Use /api use <name> to get code snippets");
					
					ctx.ui.showPanel("Random API", lines.join("\n"));
					break;
				}
				
				case "use": {
					const name = parts.slice(1).join(" ");
					if (!name) {
						ctx.ui.notify("Usage: /api use <api-name>", "warning");
						return;
					}
					
					const api = API_DATABASE.find(a => 
						a.name.toLowerCase() === name.toLowerCase()
					);
					
					if (!api) {
						ctx.ui.notify(`API "${name}" not found. Use /api search to find APIs.`, "warning");
						return;
					}
					
					const lines = [
						`=== Code Snippets: ${api.name} ===`,
						`URL: ${api.url}`,
						"",
						"--- TypeScript ---",
						generateCodeSnippet(api, "typescript"),
						"",
						"--- Python ---",
						generateCodeSnippet(api, "python"),
						"",
						"--- cURL ---",
						generateCodeSnippet(api, "curl"),
					];
					
					ctx.ui.showPanel(`${api.name} Code`, lines.join("\n"));
					break;
				}
				
				default:
					ctx.ui.notify(`Unknown command: ${subcommand}. Use: search, category, categories, random, use`, "warning");
			}
		},
	});
}
