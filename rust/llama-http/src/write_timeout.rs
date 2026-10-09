use std::{
    future::Future,
    io::{self, IoSlice},
    pin::Pin,
    task::{Context, Poll},
    time::Duration,
};

use hyper::rt::{Read, ReadBufCursor, Write};
use tokio::time::Sleep;

pub struct WriteTimeout<I> {
    io: I,
    limit: Option<Duration>,
    deadline: Option<Pin<Box<Sleep>>>,
}

impl<I> WriteTimeout<I> {
    pub fn new(io: I, limit: Option<Duration>) -> Self {
        Self {
            io,
            limit,
            deadline: None,
        }
    }

    fn guard<T>(&mut self, cx: &mut Context<'_>, poll: Poll<io::Result<T>>) -> Poll<io::Result<T>> {
        let Some(limit) = self.limit else {
            return poll;
        };
        if poll.is_ready() {
            self.deadline = None;
            return poll;
        }
        let deadline = self
            .deadline
            .get_or_insert_with(|| Box::pin(tokio::time::sleep(limit)));
        if deadline.as_mut().poll(cx).is_ready() {
            self.deadline = None;
            return Poll::Ready(Err(io::ErrorKind::TimedOut.into()));
        }
        Poll::Pending
    }
}

impl<I: Read + Unpin> Read for WriteTimeout<I> {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: ReadBufCursor<'_>,
    ) -> Poll<io::Result<()>> {
        Pin::new(&mut self.get_mut().io).poll_read(cx, buf)
    }
}

impl<I: Write + Unpin> Write for WriteTimeout<I> {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        let poll = Pin::new(&mut this.io).poll_write(cx, buf);
        this.guard(cx, poll)
    }

    fn poll_write_vectored(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        bufs: &[IoSlice<'_>],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        let poll = Pin::new(&mut this.io).poll_write_vectored(cx, bufs);
        this.guard(cx, poll)
    }

    fn is_write_vectored(&self) -> bool {
        self.io.is_write_vectored()
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        let poll = Pin::new(&mut this.io).poll_flush(cx);
        this.guard(cx, poll)
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        let poll = Pin::new(&mut this.io).poll_shutdown(cx);
        this.guard(cx, poll)
    }
}
