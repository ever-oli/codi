---
name: public-api-integration
description: Discover and integrate with 1000+ free public APIs for any data need
---

# Public API Integration

## When to Use
When the task involves:
- Fetching external data (weather, finance, news, etc.)
- Building data pipelines or dashboards
- Creating bots that need real-time information
- Replacing paid data services with free alternatives
- Prototyping apps that need API integrations

## Quick Reference: Top Free APIs by Category

### 📊 Finance & Crypto (No API Key)
- **CoinGecko** - `https://api.coingecko.com/api/v3/` - Crypto prices, market data
- **ExchangeRate-API** - `https://open.er-api.com/v6/latest/USD` - Currency exchange rates
- **FRED** - Economic data (requires free key)
- **Polymarket Gamma** - Prediction market data (read-only, no auth)

### 🌤️ Weather (No API Key)
- **Open-Meteo** - `https://api.open-meteo.com/v1/forecast` - Weather forecasts
- **wttr.in** - `https://wttr.in/{city}?format=j1` - Simple weather JSON
- **WeatherAPI** - Free tier available

### 📰 News (Free Tier)
- **GNews** - `https://gnews.io/api/v4/search` - News headlines
- **NewsAPI** - `https://newsapi.org/v2/top-headlines` - Breaking news
- **Currents API** - `https://api.currentsapi.services/v1/latest-news`

### 📍 Geolocation (No API Key)
- **ip-api.com** - `http://ip-api.com/json/{ip}` - IP geolocation
- **Nominatim** - `https://nominatim.openstreetmap.org/search` - Geocoding
- **REST Countries** - `https://restcountries.com/v3.1/` - Country data

### 🔧 Developer Tools (No API Key)
- **GitHub API** - `https://api.github.com/` - Repos, users, code
- **JSONPlaceholder** - `https://jsonplaceholder.typicode.com/` - Test data
- **httpbin** - `https://httpbin.org/` - HTTP testing
- **RandomUser** - `https://randomuser.me/api/` - Test user data

### 🎮 Entertainment (No API Key)
- **Pokémon API** - `https://pokeapi.co/api/v2/` - Pokémon data
- **Deck of Cards** - `https://deckofcardsapi.com/api/deck/new/shuffle/` - Card games
- **Trivia DB** - `https://opentdb.com/api.php` - Trivia questions

## Approach

### 1. Find the Right API
```bash
# Search by category or need
grep -i "weather\|finance\|news" ~/.pi/skills/public-api-integration/apis.json
```

### 2. Check Authentication Requirements
- `No` = Use immediately, no setup
- `apiKey` = Free signup required
- `OAuth` = More complex setup

### 3. Make the Request
```typescript
// No auth example
const response = await fetch('https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd');
const data = await response.json();

// API key example
const response = await fetch(`https://newsapi.org/v2/top-headlines?country=us&apiKey=${API_KEY}`);
```

### 4. Handle Common Patterns
```typescript
// Retry with backoff for rate limits
async function fetchWithRetry(url: string, retries = 3): Promise<Response> {
  for (let i = 0; i < retries; i++) {
    const res = await fetch(url);
    if (res.status === 429) {
      await new Promise(r => setTimeout(r, 1000 * (i + 1)));
      continue;
    }
    return res;
  }
  throw new Error('Rate limited');
}

// Cache responses
const cache = new Map();
async function cachedFetch(url: string, ttlMs = 60000) {
  const cached = cache.get(url);
  if (cached && Date.now() - cached.time < ttlMs) return cached.data;
  const data = await fetch(url).then(r => r.json());
  cache.set(url, { data, time: Date.now() });
  return data;
}
```

## API Selection Decision Tree

```
Need data type?
├── Crypto/Stocks → CoinGecko, Alpha Vantage
├── Weather → Open-Meteo (free, no key)
├── News → GNews (free tier), NewsAPI
├── Geocoding → Nominatim (OpenStreetMap)
├── IP Location → ip-api.com (free)
├── Currency → ExchangeRate-API (free)
├── Test Data → JSONPlaceholder, RandomUser
├── Images → Unsplash, Pexels (free tier)
└── Government → Many .gov APIs are free
```

## Common Pitfalls

1. **Rate Limits** - Always implement backoff; most free APIs have limits
2. **CORS** - Server-side requests work better than browser for many APIs
3. **API Keys in Code** - Use environment variables, never commit keys
4. **Data Freshness** - Check cache headers; some data updates slowly
5. **HTTPS vs HTTP** - Prefer HTTPS; some older APIs only support HTTP

## Full API Database

The complete list of 1000+ APIs is maintained at:
https://github.com/public-apis/public-apis

Categories include: Animals, Anime, Anti-Malware, Art, Authentication, 
Blockchain, Books, Business, Calendar, Cloud Storage, CI, Cryptocurrency, 
Currency, Data Validation, Development, Dictionaries, Documents, Email, 
Entertainment, Environment, Events, Finance, Food, Games, Geocoding, 
Government, Health, Jobs, Machine Learning, Music, News, Open Data, 
Patents, Phone, Photography, Programming, Science, Security, Shopping, 
Social, Sports, Test Data, Text Analysis, Tracking, Transportation, 
URL Shorteners, Vehicle, Video, Weather, and more.
