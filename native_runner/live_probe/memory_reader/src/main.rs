use std::env;
use std::fmt::Write as FmtWrite;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::os::unix::fs::FileExt;
use std::process;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const PACKAGE: &str = "nullsroyale.rel.free";
const TOKEN: &str = "nullsroyale-owner-ladder-2026-11-23_2026-12-01";
const AUTHORIZATION_START_UTC: u64 = 1_795_388_400;
const AUTHORIZATION_END_UTC_EXCLUSIVE: u64 = 1_796_166_000;
const DEFAULT_PORT: u16 = 26_792;
const MAX_READ_BYTES: usize = 4 * 1024 * 1024;
const MAX_READ_RANGES: usize = 1_024;
const MAX_COMMAND_BYTES: usize = 64 * 1024;
const MAX_SCAN_BYTES: u64 = 256 * 1024 * 1024;
const MAX_SCAN_CANDIDATES: usize = 4_096;
const SCAN_CHUNK_BYTES: usize = 4 * 1024 * 1024;
const GAME_STATE_BYTES: usize = 0x228;
const MIN_GAME_STATE_SIZE_CLASS_BYTES: u64 = 32 * 1024 * 1024;
const GAME_STATE_SIZE_CLASS_TAIL_BYTES: u64 = 20 * 1024 * 1024;

#[derive(Clone)]
struct Mapping {
    start: u64,
    end: u64,
    readable: bool,
    writable: bool,
    name: String,
}

struct Reader {
    pid: u32,
    memory: File,
}

fn now_epoch() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_secs()
}

fn authorized() -> bool {
    let now = now_epoch();
    (AUTHORIZATION_START_UTC..AUTHORIZATION_END_UTC_EXCLUSIVE).contains(&now)
}

fn exact_process(pid: u32) -> io::Result<bool> {
    let command = fs::read(format!("/proc/{pid}/cmdline"))?;
    Ok(command.split(|byte| *byte == 0).next() == Some(PACKAGE.as_bytes()))
}

fn running_as_root() -> io::Result<bool> {
    let status = fs::read_to_string("/proc/self/status")?;
    let Some(line) = status.lines().find(|line| line.starts_with("Uid:")) else {
        return Ok(false);
    };
    Ok(line
        .split_ascii_whitespace()
        .nth(1)
        .and_then(|value| value.parse::<u32>().ok())
        == Some(0))
}

fn mappings(pid: u32) -> io::Result<Vec<Mapping>> {
    let text = fs::read_to_string(format!("/proc/{pid}/maps"))?;
    let mut result = Vec::new();
    for line in text.lines() {
        let mut fields = line.split_ascii_whitespace();
        let Some(range) = fields.next() else { continue };
        let Some(perms) = fields.next() else { continue };
        let _offset = fields.next();
        let _device = fields.next();
        let _inode = fields.next();
        let name = fields.collect::<Vec<_>>().join(" ");
        let Some((start, end)) = range.split_once('-') else {
            continue;
        };
        let (Ok(start), Ok(end)) = (u64::from_str_radix(start, 16), u64::from_str_radix(end, 16))
        else {
            continue;
        };
        if start < end {
            result.push(Mapping {
                start,
                end,
                readable: perms.as_bytes().first() == Some(&b'r'),
                writable: perms.as_bytes().get(1) == Some(&b'w'),
                name,
            });
        }
    }
    Ok(result)
}

fn scan_mapping(mapping: &Mapping) -> bool {
    mapping.readable
        && mapping.writable
        && mapping.name == "[anon:scudo:primary]"
        && mapping.end - mapping.start >= MIN_GAME_STATE_SIZE_CLASS_BYTES
}

fn little_u64(data: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(data[offset..offset + 8].try_into().expect("bounded u64"))
}

fn little_u32(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(data[offset..offset + 4].try_into().expect("bounded u32"))
}

fn permitted(maps: &[Mapping], address: u64, size: usize) -> bool {
    let Ok(size) = u64::try_from(size) else {
        return false;
    };
    let Some(end) = address.checked_add(size) else {
        return false;
    };
    size > 0
        && maps
            .iter()
            .any(|mapping| mapping.readable && mapping.start <= address && end <= mapping.end)
}

fn parse_address(value: &str) -> Option<u64> {
    let value = value.strip_prefix("0x").unwrap_or(value);
    u64::from_str_radix(value, 16).ok()
}

fn parse_range(value: &str) -> Option<(u64, usize)> {
    let (address, size) = value.split_once(':')?;
    Some((parse_address(address)?, size.parse::<usize>().ok()?))
}

fn write_error(stream: &mut TcpStream, message: &str) -> io::Result<()> {
    let safe = message.replace(['\r', '\n', '"'], " ");
    writeln!(stream, "ERR {safe}")?;
    stream.flush()
}

fn read_exact_at(memory: &File, address: u64, size: usize) -> io::Result<Vec<u8>> {
    let mut output = vec![0_u8; size];
    let mut used = 0;
    while used < size {
        let count = memory.read_at(&mut output[used..], address + used as u64)?;
        if count == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "process memory returned an early EOF",
            ));
        }
        used += count;
    }
    Ok(output)
}

impl Reader {
    fn validate(&self) -> io::Result<Vec<Mapping>> {
        if !authorized() {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "authorization window is closed",
            ));
        }
        if !exact_process(self.pid)? {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "target PID no longer belongs to the exact game package",
            ));
        }
        mappings(self.pid)
    }

    fn read_ranges(&self, ranges: &[(u64, usize)]) -> io::Result<Vec<Vec<u8>>> {
        if ranges.is_empty() || ranges.len() > MAX_READ_RANGES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid read range count",
            ));
        }
        let total = ranges.iter().try_fold(0_usize, |total, (_, size)| {
            total.checked_add(*size).ok_or(())
        });
        if total.is_err() || total.unwrap_or(usize::MAX) > MAX_READ_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "read exceeds the fixed byte limit",
            ));
        }
        let maps = self.validate()?;
        if ranges
            .iter()
            .any(|(address, size)| !permitted(&maps, *address, *size))
        {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "read range is not contained in a readable mapping",
            ));
        }
        ranges
            .iter()
            .map(|(address, size)| read_exact_at(&self.memory, *address, *size))
            .collect()
    }

    fn scan_game_states(&self) -> io::Result<Vec<(u64, u64)>> {
        let maps = self.validate()?;
        let largest_size_class = maps
            .iter()
            .filter(|mapping| scan_mapping(mapping))
            .max_by_key(|mapping| mapping.end - mapping.start)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    "current build exposes no GameState Scudo size class",
                )
            })?;
        let scan_bytes = (largest_size_class.end - largest_size_class.start)
            .min(GAME_STATE_SIZE_CLASS_TAIL_BYTES);
        if scan_bytes > MAX_SCAN_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "candidate scan exceeds its fixed byte bound",
            ));
        }

        let mut candidates = Vec::new();
        let scan_start = largest_size_class.end - scan_bytes;
        let mut address = scan_start;
        while address < largest_size_class.end {
            let remaining = largest_size_class.end - address;
            let size = usize::try_from(remaining.min((SCAN_CHUNK_BYTES + GAME_STATE_BYTES) as u64))
                .expect("bounded scan chunk");
            let payload = match read_exact_at(&self.memory, address, size) {
                Ok(payload) => payload,
                Err(_) => break,
            };
            let first = usize::try_from((16 - address % 16) % 16).expect("alignment offset");
            let exclusive = payload.len().saturating_sub(GAME_STATE_BYTES - 1);
            for offset in (first..exclusive).step_by(16) {
                let state = little_u32(&payload, offset + 0x20);
                let tick = little_u32(&payload, offset + 0x60);
                let guard = payload[offset + 0x1e8];
                if state > 16 || tick > 200_000 || guard > 1 {
                    continue;
                }
                let owned = [0x08, 0x48, 0x1d8, 0x210, 0x218]
                    .iter()
                    .all(|field| permitted(&maps, little_u64(&payload, offset + field), 1));
                let world = little_u64(&payload, offset + 0xa8);
                if !owned || !permitted(&maps, world, 0x200) {
                    continue;
                }
                candidates.push((address + offset as u64, world));
                if candidates.len() > MAX_SCAN_CANDIDATES {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "GameState candidate scan exceeded its ambiguity bound",
                    ));
                }
            }
            if remaining <= SCAN_CHUNK_BYTES as u64 {
                break;
            }
            address += SCAN_CHUNK_BYTES as u64;
        }
        Ok(candidates)
    }
}

fn handle_command(stream: &mut TcpStream, reader: &Reader, command: &str) -> io::Result<bool> {
    let fields: Vec<&str> = command.split_ascii_whitespace().collect();
    let Some(verb) = fields.first().copied() else {
        write_error(stream, "empty command")?;
        return Ok(false);
    };
    let token_ok = fields.get(1).copied() == Some(TOKEN);
    if !token_ok {
        write_error(stream, "authorization token mismatch")?;
        return Ok(false);
    }
    match verb {
        "status" if fields.len() == 2 => {
            let process_ok = exact_process(reader.pid).unwrap_or(false);
            writeln!(
                stream,
                "OK {{\"schema\":\"nullsroyale-external-reader.v1\",\"pid\":{},\"package\":\"{}\",\"readOnly\":true,\"authorized\":{},\"processMatches\":{},\"authorizationEndUtcExclusive\":{}}}",
                reader.pid,
                PACKAGE,
                authorized(),
                process_ok,
                AUTHORIZATION_END_UTC_EXCLUSIVE
            )?;
            stream.flush()?;
        }
        "maps" if fields.len() == 2 => {
            reader.validate()?;
            let payload = fs::read(format!("/proc/{}/maps", reader.pid))?;
            writeln!(stream, "OK {}", payload.len())?;
            stream.write_all(&payload)?;
            stream.flush()?;
        }
        "read" if fields.len() == 4 => {
            let Some(address) = parse_address(fields[2]) else {
                write_error(stream, "invalid hexadecimal address")?;
                return Ok(false);
            };
            let Ok(size) = fields[3].parse::<usize>() else {
                write_error(stream, "invalid read size")?;
                return Ok(false);
            };
            let payloads = reader.read_ranges(&[(address, size)])?;
            writeln!(stream, "OK {size}")?;
            stream.write_all(&payloads[0])?;
            stream.flush()?;
        }
        "readv" if fields.len() >= 3 => {
            let ranges: Option<Vec<(u64, usize)>> =
                fields[2..].iter().map(|value| parse_range(value)).collect();
            let Some(ranges) = ranges else {
                write_error(stream, "invalid readv range")?;
                return Ok(false);
            };
            let payloads = reader.read_ranges(&ranges)?;
            let total: usize = payloads.iter().map(Vec::len).sum();
            writeln!(stream, "OK {total}")?;
            for payload in payloads {
                stream.write_all(&payload)?;
            }
            stream.flush()?;
        }
        "scan-gamestate" if fields.len() == 2 => {
            let candidates = reader.scan_game_states()?;
            let mut payload =
                String::from("{\"schema\":\"nullsroyale-gamestate-scan.v1\",\"candidates\":[");
            for (index, (manager, world)) in candidates.iter().enumerate() {
                if index > 0 {
                    payload.push(',');
                }
                write!(&mut payload, "[{manager},{world}]").expect("write to String");
            }
            payload.push_str("]}");
            writeln!(stream, "OK {}", payload.len())?;
            stream.write_all(payload.as_bytes())?;
            stream.flush()?;
        }
        "quit" if fields.len() == 2 => return Ok(true),
        _ => write_error(
            stream,
            "supported commands are status, maps, read, readv, scan-gamestate and quit",
        )?,
    }
    Ok(false)
}

fn handle_client(stream: TcpStream, reader: &Reader) -> io::Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(10)))?;
    stream.set_write_timeout(Some(Duration::from_secs(10)))?;
    let mut output = stream.try_clone()?;
    let mut input = BufReader::new(stream);
    loop {
        let mut command = String::new();
        let count = input.read_line(&mut command)?;
        if count == 0 {
            return Ok(());
        }
        if command.len() > MAX_COMMAND_BYTES {
            write_error(&mut output, "command exceeds fixed size limit")?;
            return Ok(());
        }
        match handle_command(&mut output, reader, command.trim()) {
            Ok(true) => return Ok(()),
            Ok(false) => {}
            Err(error) => write_error(&mut output, &error.to_string())?,
        }
    }
}

fn parse_args() -> Result<(u32, u16), String> {
    let mut args = env::args().skip(1);
    let pid = args
        .next()
        .ok_or("usage: nullsroyale-memory-reader PID [PORT]")?
        .parse::<u32>()
        .map_err(|_| "PID must be a positive decimal integer")?;
    if pid == 0 {
        return Err("PID must be positive".into());
    }
    let port = args
        .next()
        .map(|value| {
            value
                .parse::<u16>()
                .map_err(|_| "PORT must be a decimal u16")
        })
        .transpose()?
        .unwrap_or(DEFAULT_PORT);
    if args.next().is_some() || port == 0 {
        return Err("usage: nullsroyale-memory-reader PID [PORT]".into());
    }
    Ok((pid, port))
}

fn run() -> Result<(), String> {
    let (pid, port) = parse_args()?;
    if !running_as_root().map_err(|error| error.to_string())? {
        return Err("reader must run as root".into());
    }
    if !authorized() {
        return Err("authorization window is closed".into());
    }
    if !exact_process(pid).map_err(|error| error.to_string())? {
        return Err(format!("PID {pid} is not the exact {PACKAGE} process"));
    }
    let memory = File::open(format!("/proc/{pid}/mem")).map_err(|error| error.to_string())?;
    let listener = TcpListener::bind(("127.0.0.1", port)).map_err(|error| error.to_string())?;
    let reader = Reader { pid, memory };
    eprintln!("Null's Royale external reader: pid={pid} loopback=127.0.0.1:{port} read-only=true");
    for incoming in listener.incoming() {
        match incoming {
            Ok(stream) => {
                if let Err(error) = handle_client(stream, &reader) {
                    eprintln!("client error: {error}");
                }
            }
            Err(error) => eprintln!("accept error: {error}"),
        }
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("Null's Royale external reader refused to start: {error}");
        process::exit(2);
    }
}
