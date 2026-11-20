# Generic trampoline for the guarded-stack test.  Places exactly the right number of stack
# words on a stack whose upper end is an unreadable page, sets up integer and SSE registers
# from a descriptor struct, and calls the function.  Any read past the declared stack words
# hits the guard page and raises SIGSEGV.
#
# int call_at_guard(struct call_desc *desc);
#
# struct call_desc {
#     void *guard;          /*  0  first byte of the unreadable page              */
#     void *fn;             /*  8  the function to call                           */
#     long  n_stack;        /* 16  number of 8-byte stack words (always even)     */
#     long *stack_data;     /* 24  array of n_stack longs to place on the stack   */
#     long  ireg[6];        /* 32  rdi, rsi, rdx, rcx, r8, r9                    */
#     double sreg[8];       /* 80  xmm0 through xmm7                             */
# };                        /* total 144 bytes                                    */

    .text
    .globl call_at_guard
    .type  call_at_guard,@function
call_at_guard:
    push   %rbp
    mov    %rsp,%rbp
    push   %rbx
    push   %r12
    push   %r13
    push   %r14
    push   %r15

    mov    %rdi,%r12            # desc

    # Build the new stack.  rsp = guard - 8*n_stack (16-byte aligned: n is even, guard is
    # page-aligned).  Copy the stack_data[] words there.
    mov    0(%r12),%r13         # guard
    mov    16(%r12),%r15        # n_stack
    mov    24(%r12),%rbx        # stack_data
    lea    (,%r15,8),%rax
    sub    %rax,%r13            # r13 = new rsp before the call
    mov    %r13,%rsp            # switch to the guarded stack

    xor    %ecx,%ecx
1:  cmp    %r15,%rcx
    jge    2f
    mov    (%rbx,%rcx,8),%rax
    mov    %rax,(%rsp,%rcx,8)
    inc    %rcx
    jmp    1b
2:

    # Load SSE registers from desc->sreg[].
    movsd  80(%r12),%xmm0
    movsd  88(%r12),%xmm1
    movsd  96(%r12),%xmm2
    movsd  104(%r12),%xmm3
    movsd  112(%r12),%xmm4
    movsd  120(%r12),%xmm5
    movsd  128(%r12),%xmm6
    movsd  136(%r12),%xmm7

    # Load integer registers from desc->ireg[] and call the function.
    mov    8(%r12),%r14         # fn (must load before we clobber r14 below)
    mov    32(%r12),%rdi
    mov    40(%r12),%rsi
    mov    48(%r12),%rdx
    mov    56(%r12),%rcx
    mov    64(%r12),%r8
    mov    72(%r12),%r9
    call   *%r14

    # Return value is already in eax.  Restore the original stack: rbp points to the saved
    # rbp; the five callee-saved registers were pushed after it, so the saved r15 is at
    # rbp - 40.
    lea    -40(%rbp),%rsp
    pop    %r15
    pop    %r14
    pop    %r13
    pop    %r12
    pop    %rbx
    pop    %rbp
    ret
    .size  call_at_guard,.-call_at_guard

.section .note.GNU-stack,"",@progbits
