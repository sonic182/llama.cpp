use hyper::Method;

use crate::{
    abi::{METHOD_DELETE, METHOD_GET, METHOD_POST},
    codec::{self, Pair},
};

#[derive(Default)]
pub struct Routes {
    get: matchit::Router<usize>,
    post: matchit::Router<usize>,
    delete: matchit::Router<usize>,
}

fn to_matchit(path: &str) -> String {
    path.replace('{', "{{")
        .replace('}', "}}")
        .split('/')
        .map(|segment| match segment.strip_prefix(':') {
            Some(name) => format!("{{{name}}}"),
            None => segment.to_owned(),
        })
        .collect::<Vec<_>>()
        .join("/")
}

impl Routes {
    pub fn add(&mut self, method: i32, path: &str, id: usize) -> Result<(), String> {
        let router = match method {
            METHOD_GET => &mut self.get,
            METHOD_POST => &mut self.post,
            METHOD_DELETE => &mut self.delete,
            other => return Err(format!("unknown method {other}")),
        };
        router
            .insert(to_matchit(path), id)
            .map_err(|e| e.to_string())
    }

    pub fn find(&self, method: &Method, path: &str) -> Option<(usize, Vec<Pair>)> {
        let router = match *method {
            Method::GET | Method::HEAD => &self.get,
            Method::POST => &self.post,
            Method::DELETE => &self.delete,
            _ => return None,
        };
        let matched = router.at(path).ok()?;
        let params = matched
            .params
            .iter()
            .map(|(key, value)| {
                (
                    key.as_bytes().to_vec(),
                    codec::decode(value.as_bytes(), false),
                )
            })
            .collect();
        Some((*matched.value, params))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_colon_params_and_decodes_them() {
        let mut routes = Routes::default();
        routes.add(METHOD_POST, "/slots/:id_slot", 7).unwrap();
        routes.add(METHOD_GET, "/health", 1).unwrap();

        let (id, params) = routes.find(&Method::POST, "/slots/conv%3A%3A1").unwrap();
        assert_eq!(id, 7);
        assert_eq!(params, [(b"id_slot".to_vec(), b"conv::1".to_vec())]);

        assert!(routes.find(&Method::GET, "/slots/1").is_none());
        assert_eq!(routes.find(&Method::HEAD, "/health").unwrap().0, 1);
        assert!(routes.find(&Method::GET, "/health/").is_none());
    }
}
